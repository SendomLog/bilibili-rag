import unittest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import (
    Base,
    ContentSource,
    FavoriteFolder,
    FavoriteVideo,
    UserSession,
    VideoCache,
    VideoContent,
)
from app.routers.knowledge import (
    MarkdownExportRequest,
    _ingest_single_video,
    _start_operation,
    active_operations,
    cancel_operation,
    export_video_markdown,
)
from app.services.cancellation import OperationCancelled
from app.services.markdown_export import (
    build_video_markdown,
    organize_video_content,
    split_content,
)


def _completion(content: str) -> Mock:
    response = Mock()
    response.choices = [Mock(message=Mock(content=content))]
    return response


class MarkdownExportTest(unittest.TestCase):
    def test_original_export_keeps_metadata_and_full_content(self):
        video = VideoCache(
            bvid="BV1",
            title="测试视频",
            description="视频简介",
            owner_name="作者",
            content_source="asr",
            content="第一段原文\n第二段原文",
        )

        markdown = build_video_markdown(
            video,
            exported_at=datetime(2026, 6, 13, 10, 30, 0),
        )

        self.assertIn('title: "测试视频"', markdown)
        self.assertIn("https://www.bilibili.com/video/BV1", markdown)
        self.assertIn("## 视频简介\n\n视频简介", markdown)
        self.assertIn("## 原始内容\n\n第一段原文\n第二段原文", markdown)
        self.assertNotIn("## AI 内容整理", markdown)

    def test_ai_export_adds_summary_without_replacing_original(self):
        video = VideoCache(bvid="BV1", title="测试视频", content="完整原文")

        markdown = build_video_markdown(video, ai_content="### 内容摘要\n\n整理结果")

        self.assertIn("## AI 内容整理\n\n### 内容摘要\n\n整理结果", markdown)
        self.assertIn("## 原始内容\n\n完整原文", markdown)

    def test_empty_content_cannot_be_exported(self):
        with self.assertRaisesRegex(ValueError, "尚无可导出"):
            build_video_markdown(VideoCache(bvid="BV1", title="测试", content="  "))

    def test_long_content_is_summarized_in_chunks_before_final_merge(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            _completion("片段一笔记"),
            _completion("片段二笔记"),
            _completion("最终整理"),
        ]

        result = organize_video_content(
            "长视频",
            "A" * 12,
            client=client,
            model="test-model",
            chunk_size=6,
        )

        self.assertEqual(result, "最终整理")
        self.assertEqual(client.chat.completions.create.call_count, 3)
        final_messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("片段一笔记", final_messages[1]["content"])
        self.assertIn("片段二笔记", final_messages[1]["content"])

    def test_empty_ai_response_is_not_silently_exported(self):
        client = Mock()
        client.chat.completions.create.return_value = _completion("  ")

        with self.assertRaisesRegex(RuntimeError, "AI 内容整理返回空结果"):
            organize_video_content("测试视频", "完整原文", client=client, model="test-model")

    def test_ai_export_cancellation_stops_following_model_requests(self):
        cancelled = False
        client = Mock()

        def complete_first_chunk(**_kwargs):
            nonlocal cancelled
            cancelled = True
            return _completion("片段一笔记")

        client.chat.completions.create.side_effect = complete_first_chunk

        with self.assertRaises(OperationCancelled):
            organize_video_content(
                "长视频",
                "A" * 12,
                client=client,
                model="test-model",
                chunk_size=6,
                cancel_check=lambda: cancelled,
            )

        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_split_content_prefers_newline_boundary(self):
        self.assertEqual(split_content("1234\n5678", chunk_size=6), ["1234", "5678"])


class MarkdownExportEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def asyncTearDown(self):
        active_operations.clear()
        await self.engine.dispose()

    async def test_cancel_operation_sets_server_cancel_event(self):
        event = _start_operation("operation-1", "session")

        with patch("app.routers.knowledge.get_session", new=AsyncMock(return_value={"cookies": {}})):
            response = await cancel_operation("operation-1", "session")

        self.assertTrue(event.is_set())
        self.assertEqual(response["message"], "取消请求已发送")

    async def test_export_allows_same_user_historical_session_only(self):
        async with self.session_factory() as db:
            folder = FavoriteFolder(session_id="owner-session", media_id=1, title="folder")
            video = VideoCache(bvid="BV1", title="测试视频", content="完整原文")
            db.add_all([
                folder,
                video,
                UserSession(session_id="owner-session", bili_mid=1),
                UserSession(session_id="same-user-session", bili_mid=1),
                UserSession(session_id="other-session", bili_mid=2),
            ])
            await db.flush()
            db.add(FavoriteVideo(folder_id=folder.id, bvid=video.bvid))
            await db.commit()

            with patch("app.routers.knowledge.get_session", new=AsyncMock(return_value={"cookies": {}})):
                response = await export_video_markdown(
                    "BV1",
                    MarkdownExportRequest(mode="original"),
                    "owner-session",
                    db,
                )
                self.assertIn("完整原文", response.body.decode())

                response = await export_video_markdown(
                    "BV1",
                    MarkdownExportRequest(mode="original"),
                    "same-user-session",
                    db,
                )
                self.assertIn("完整原文", response.body.decode())

                with self.assertRaises(HTTPException) as error:
                    await export_video_markdown(
                        "BV1",
                        MarkdownExportRequest(mode="original"),
                        "other-session",
                        db,
                    )

        self.assertEqual(error.exception.status_code, 404)

    async def test_single_video_ingest_only_adds_requested_video(self):
        async with self.session_factory() as db:
            bili = Mock()
            bili.get_favorite_content = AsyncMock(
                return_value={"info": {"title": "folder", "media_count": 2}}
            )
            bili.get_all_favorite_videos = AsyncMock(
                return_value=[
                    {"bvid": "BV1", "title": "目标视频", "cid": 1},
                    {"bvid": "BV2", "title": "其他视频", "cid": 2},
                ]
            )
            content_fetcher = Mock()
            content_fetcher.fetch_content = AsyncMock(
                return_value=VideoContent(
                    bvid="BV1",
                    title="目标视频",
                    content="有效字幕内容" * 20,
                    source=ContentSource.ASR,
                )
            )
            rag = Mock()
            rag.has_video.return_value = False
            rag.add_video_content.return_value = 2

            cache = await _ingest_single_video(
                db,
                bili,
                rag,
                content_fetcher,
                "session",
                1,
                "BV1",
            )
            relations = await db.scalars(select(FavoriteVideo.bvid))

        self.assertEqual(cache.bvid, "BV1")
        self.assertTrue(cache.is_processed)
        self.assertEqual(list(relations), ["BV1"])
        content_fetcher.fetch_content.assert_awaited_once()
        rag.add_video_content.assert_called_once()

    async def test_single_video_ingest_cancellation_rolls_back_before_vector_write(self):
        async with self.session_factory() as db:
            cancelled = False
            bili = Mock()
            bili.get_favorite_content = AsyncMock(
                return_value={"info": {"title": "folder", "media_count": 1}}
            )
            bili.get_all_favorite_videos = AsyncMock(
                return_value=[{"bvid": "BV1", "title": "目标视频", "cid": 1}]
            )
            content_fetcher = Mock()

            async def fetch_content(*_args, **_kwargs):
                nonlocal cancelled
                cancelled = True
                return VideoContent(
                    bvid="BV1",
                    title="目标视频",
                    content="有效字幕内容" * 20,
                    source=ContentSource.ASR,
                )

            content_fetcher.fetch_content.side_effect = fetch_content
            rag = Mock()
            rag.has_video.return_value = False

            with self.assertRaises(OperationCancelled):
                await _ingest_single_video(
                    db,
                    bili,
                    rag,
                    content_fetcher,
                    "session",
                    1,
                    "BV1",
                    cancel_check=lambda: cancelled,
                )

            relations = await db.scalars(select(FavoriteVideo.bvid))

        self.assertEqual(list(relations), [])
        rag.add_video_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
