import unittest
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from langchain.schema import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Base, ContentSource, FavoriteFolder, FavoriteVideo, VideoCache, VideoContent
from app.routers.knowledge import _set_cache_processing_result, _sync_folder, get_folder_status
from app.services.rag import RAGService


class FailureVisibilityTest(unittest.TestCase):
    def test_cache_records_vector_failure_and_can_be_cleared(self):
        cache = VideoCache(bvid="BV1", title="test", is_processed=True)

        _set_cache_processing_result(cache, RuntimeError("embedding unavailable"))

        self.assertFalse(cache.is_processed)
        self.assertEqual(cache.process_error, "embedding unavailable")

        _set_cache_processing_result(cache)

        self.assertTrue(cache.is_processed)
        self.assertIsNone(cache.process_error)

    def test_vector_search_failure_is_not_returned_as_empty_results(self):
        rag = RAGService.__new__(RAGService)
        rag.vectorstore = Mock()
        rag.vectorstore.max_marginal_relevance_search.side_effect = RuntimeError("mmr failed")
        rag.vectorstore.similarity_search.side_effect = RuntimeError("embedding failed")

        with self.assertRaisesRegex(RuntimeError, "向量检索失败"):
            rag.search("test")

    def test_mmr_failure_still_falls_back_to_similarity_search(self):
        rag = RAGService.__new__(RAGService)
        rag.vectorstore = Mock()
        rag.vectorstore.max_marginal_relevance_search.side_effect = RuntimeError("mmr failed")
        docs = [Document(page_content="doc", metadata={})]
        rag.vectorstore.similarity_search.return_value = docs

        self.assertEqual(rag.search("test"), docs)

    def test_has_video_raises_when_chroma_query_fails(self):
        rag = RAGService.__new__(RAGService)
        rag.vectorstore = Mock()
        rag.vectorstore._collection.get.side_effect = RuntimeError("chroma unavailable")

        with self.assertRaisesRegex(RuntimeError, "查询视频向量失败"):
            rag.has_video("BV1")

    def test_partial_vector_write_is_cleaned_up(self):
        rag = RAGService.__new__(RAGService)
        rag.text_splitter = Mock()
        rag.text_splitter.split_text.return_value = [f"chunk-{index}" for index in range(11)]
        rag._build_metadata_document = Mock(return_value=None)
        rag.vectorstore = Mock()
        rag.vectorstore.add_documents.side_effect = [
            [f"id-{index}" for index in range(10)],
            RuntimeError("second batch failed"),
        ]
        video = VideoContent(
            bvid="BV1",
            title="video",
            content="有效内容" * 5,
            source=ContentSource.ASR,
        )

        with self.assertRaisesRegex(RuntimeError, "second batch failed"):
            rag.add_video_content(video)

        rag.vectorstore._collection.delete.assert_called_once_with(
            ids=[f"id-{index}" for index in range(10)]
        )


class SyncFailureVisibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_vector_failure_is_recorded_and_not_counted_as_indexed(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            bili = Mock()
            bili.get_favorite_content = AsyncMock(
                return_value={"info": {"title": "folder", "media_count": 1}}
            )
            bili.get_all_favorite_videos = AsyncMock(
                return_value=[{"bvid": "BV1", "title": "video", "cid": 1}]
            )
            content_fetcher = Mock()
            content_fetcher.fetch_content = AsyncMock(
                return_value=VideoContent(
                    bvid="BV1",
                    title="video",
                    content="有效字幕内容" * 20,
                    source=ContentSource.ASR,
                )
            )
            rag = Mock()
            rag.add_video_content.side_effect = RuntimeError("embedding unavailable")

            result = await _sync_folder(
                db,
                bili,
                rag,
                content_fetcher,
                session_id="session",
                folder_id=1,
            )

            cache = await db.scalar(select(VideoCache).where(VideoCache.bvid == "BV1"))
            relation = await db.scalar(select(FavoriteVideo).where(FavoriteVideo.bvid == "BV1"))

        await engine.dispose()

        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["indexed"], 0)
        self.assertIsNotNone(relation)
        self.assertFalse(cache.is_processed)
        self.assertEqual(cache.process_error, "embedding unavailable")

    async def test_missing_historical_vectors_are_rebuilt(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            folder = FavoriteFolder(session_id="session", media_id=1, title="folder", media_count=1)
            cache = VideoCache(
                bvid="BV1",
                title="video",
                content="有效字幕内容" * 20,
                content_source=ContentSource.ASR.value,
                is_processed=True,
            )
            db.add_all([folder, cache])
            await db.flush()
            db.add(FavoriteVideo(folder_id=folder.id, bvid="BV1"))
            await db.commit()

            bili = Mock()
            bili.get_favorite_content = AsyncMock(
                return_value={"info": {"title": "folder", "media_count": 1}}
            )
            bili.get_all_favorite_videos = AsyncMock(
                return_value=[{"bvid": "BV1", "title": "video", "cid": 1}]
            )
            rag = Mock()
            rag.has_video.return_value = False
            rag.add_video_content.return_value = 1

            result = await _sync_folder(db, bili, rag, Mock(), "session", 1)

            cache = await db.scalar(select(VideoCache).where(VideoCache.bvid == "BV1"))

        await engine.dispose()

        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["indexed"], 1)
        rag.add_video_content.assert_called_once()
        self.assertTrue(cache.is_processed)
        self.assertIsNone(cache.process_error)

    async def test_existing_vectors_are_not_rebuilt(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            folder = FavoriteFolder(session_id="session", media_id=1, title="folder", media_count=1)
            cache = VideoCache(
                bvid="BV1",
                title="video",
                content="有效字幕内容" * 20,
                content_source=ContentSource.ASR.value,
                is_processed=True,
            )
            db.add_all([folder, cache])
            await db.flush()
            db.add(FavoriteVideo(folder_id=folder.id, bvid="BV1"))
            await db.commit()

            bili = Mock()
            bili.get_favorite_content = AsyncMock(
                return_value={"info": {"title": "folder", "media_count": 1}}
            )
            bili.get_all_favorite_videos = AsyncMock(
                return_value=[{"bvid": "BV1", "title": "video", "cid": 1}]
            )
            rag = Mock()
            rag.has_video.return_value = True

            result = await _sync_folder(db, bili, rag, Mock(), "session", 1)

        await engine.dispose()

        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["indexed"], 1)
        rag.add_video_content.assert_not_called()

    async def test_nonempty_folder_returning_no_videos_fails_sync(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            bili = Mock()
            bili.get_favorite_content = AsyncMock(
                return_value={"info": {"title": "folder", "media_count": 1}}
            )
            bili.get_all_favorite_videos = AsyncMock(return_value=[])

            with self.assertRaisesRegex(RuntimeError, "已中止同步以避免误删"):
                await _sync_folder(db, bili, Mock(), Mock(), "session", 1)

        await engine.dispose()

    async def test_folder_status_calibrates_missing_vectors_and_reports_failure(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            folder = FavoriteFolder(
                session_id="session",
                media_id=1,
                title="folder",
                media_count=2,
                last_sync_at=datetime.utcnow(),
            )
            caches = [
                VideoCache(bvid="BV1", title="ok", is_processed=True),
                VideoCache(bvid="BV2", title="missing", is_processed=True),
            ]
            db.add_all([folder, *caches])
            await db.flush()
            db.add_all(
                [
                    FavoriteVideo(folder_id=folder.id, bvid="BV1"),
                    FavoriteVideo(folder_id=folder.id, bvid="BV2"),
                ]
            )
            await db.commit()

            rag = Mock()
            rag.has_video.side_effect = lambda bvid: bvid == "BV1"
            with unittest.mock.patch("app.routers.knowledge.get_rag_service", return_value=rag):
                statuses = await get_folder_status("session", db)

            missing_cache = await db.scalar(select(VideoCache).where(VideoCache.bvid == "BV2"))

        await engine.dispose()

        self.assertEqual(statuses[0].indexed_count, 1)
        self.assertEqual(statuses[0].failed_count, 1)
        self.assertFalse(missing_cache.is_processed)
        self.assertEqual(missing_cache.process_error, "向量数据缺失，等待重新入库")


if __name__ == "__main__":
    unittest.main()
