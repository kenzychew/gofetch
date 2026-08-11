"""Tests for BM25 index persistence and env-var-configurable storage location."""

from pathlib import Path

import pytest

from src.config import AppConfig
from src.exceptions import IndexingError
from src.ingestion.indexer import BM25Indexer
from src.schemas import Chunk


def test_bm25_index_dir_env_var_overrides_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BM25_INDEX_DIR should override the directory but keep the filename."""
    monkeypatch.setenv("BM25_INDEX_DIR", str(tmp_path / "mounted-volume"))

    config = AppConfig()

    assert config.bm25_index_path == str(tmp_path / "mounted-volume" / "bm25.pkl")


def test_bm25_index_dir_unset_keeps_local_dev_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With BM25_INDEX_DIR unset, the relative local-dev default is unchanged."""
    monkeypatch.delenv("BM25_INDEX_DIR", raising=False)

    config = AppConfig()

    assert config.bm25_index_path == "bm25_index/bm25.pkl"


def test_bm25_index_dir_env_var_preserves_custom_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A custom bm25_index_path's filename survives the directory override."""
    monkeypatch.setenv("BM25_INDEX_DIR", str(tmp_path / "mounted-volume"))

    config = AppConfig(bm25_index_path="custom_dir/my_index.pkl")

    assert config.bm25_index_path == str(tmp_path / "mounted-volume" / "my_index.pkl")


def test_build_index_then_load_index_roundtrip(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    """Building an index and loading it back should reproduce the same chunks."""
    indexer = BM25Indexer(str(tmp_path / "bm25_index" / "bm25.pkl"))

    indexer.build_index(sample_chunks)
    bm25, loaded_chunks = indexer.load_index()

    assert bm25 is not None
    assert [c.chunk_id for c in loaded_chunks] == [c.chunk_id for c in sample_chunks]


def test_load_index_missing_file_raises_indexing_error(tmp_path: Path) -> None:
    """Loading from a directory that exists but has no pickle yet should raise cleanly."""
    empty_dir = tmp_path / "fresh-volume"
    empty_dir.mkdir()
    indexer = BM25Indexer(str(empty_dir / "bm25.pkl"))

    with pytest.raises(IndexingError, match="not found"):
        indexer.load_index()


def test_load_index_corrupt_file_raises_indexing_error(tmp_path: Path) -> None:
    """Loading an unreadable/corrupt pickle should raise IndexingError, not crash unhandled."""
    index_path = tmp_path / "bm25_index" / "bm25.pkl"
    index_path.parent.mkdir(parents=True)
    index_path.write_bytes(b"not a valid pickle")

    indexer = BM25Indexer(str(index_path))

    with pytest.raises(IndexingError):
        indexer.load_index()
