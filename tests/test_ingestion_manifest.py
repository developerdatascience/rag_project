
from src.ingestion_manifest import IngestionManifest


def test_new_document_needs_ingestion(tmp_path):
    manifest = IngestionManifest(tmp_path / "manifest.db")
    path = tmp_path / "document.pdf"

    assert manifest.needs_ingestion(
        path, "hash_v1", 800, 100
    )


def test_successful_ingestion_is_recorded(tmp_path):
    manifest = IngestionManifest(tmp_path / "manifest.db")
    path = tmp_path / "document.pdf"

    manifest.mark_processing(
        path, "hash_v1", "doc_v1", 800, 100
    )
    manifest.mark_indexed(path)

    assert not manifest.needs_ingestion(
        path, "hash_v1", 800, 100
    )


def test_changed_content_requires_ingestion(tmp_path):
    manifest = IngestionManifest(tmp_path / "manifest.db")
    path = tmp_path / "document.pdf"

    manifest.mark_processing(
        path, "hash_v1", "doc_v1", 800, 100
    )
    manifest.mark_indexed(path)

    assert manifest.needs_ingestion(
        path, "hash_v2", 800, 100
    )


def test_changed_chunking_requires_ingestion(tmp_path):
    manifest = IngestionManifest(tmp_path / "manifest.db")
    path = tmp_path / "document.pdf"

    manifest.mark_processing(
        path, "hash_v1", "doc_v1", 800, 100
    )
    manifest.mark_indexed(path)

    assert manifest.needs_ingestion(
        path, "hash_v1", 1200, 150
    )


def test_failed_ingestion_can_be_retried(tmp_path):
    manifest = IngestionManifest(tmp_path / "manifest.db")
    path = tmp_path / "document.pdf"

    manifest.mark_processing(
        path, "hash_v1", "doc_v1", 800, 100
    )
    manifest.mark_failed(path, "Embedding provider unavailable")

    record = manifest.get(path)

    assert record["status"] == "failed"
    assert "unavailable" in record["error_message"]
    assert manifest.needs_ingestion(
        path, "hash_v1", 800, 100
    )


def test_manifest_survives_reopening(tmp_path):
    db_path = tmp_path / "manifest.db"
    path = tmp_path / "document.pdf"

    first = IngestionManifest(db_path)
    first.mark_processing(path, "hash_v1", "doc_v1", 800, 100)
    first.mark_indexed(path)

    reopened = IngestionManifest(db_path)

    assert not reopened.needs_ingestion(
        path, "hash_v1", 800, 100
    )
