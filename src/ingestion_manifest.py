"""
This is a key production RAG concept. When a user ingests the same document multiple times, 
we should not create duplicate vectors. When the document changes, the system should 
identify the new version and replace the old version's indexed chunks.

The manifest will track the source path, content hash, document ID, chunking configuration, and ingestion status. 
A failed ingestion must not be recorded as successful.
"""

import sqlite3
from pathlib import Path


class IngestionManifest:
    """Track document versions and their indexing status."""

    def __init__(self, db_path="data/ingestion_manifest.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_manifest (
                    source_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('processing', 'indexed', 'failed')),
                    error_message TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get(self, source_path):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM ingestion_manifest
                WHERE source_path = ?
                """,
                (str(Path(source_path).resolve()),),
            ).fetchone()

        return dict(row) if row else None

    def needs_ingestion(
        self,
        source_path,
        content_hash,
        chunk_size,
        chunk_overlap,
    ):
        """Return True unless this exact version is marked indexed."""
        record = self.get(source_path)

        return not (
            record
            and record["content_hash"] == content_hash
            and record["chunk_size"] == chunk_size
            and record["chunk_overlap"] == chunk_overlap
            and record["status"] == "indexed"
        )

    def mark_processing(
        self,
        source_path,
        content_hash,
        document_id,
        chunk_size,
        chunk_overlap,
    ):
        normalized_path = str(Path(source_path).resolve())

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_manifest (
                    source_path, content_hash, document_id,
                    chunk_size, chunk_overlap, status, error_message,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'processing', NULL, CURRENT_TIMESTAMP)
                ON CONFLICT(source_path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    document_id = excluded.document_id,
                    chunk_size = excluded.chunk_size,
                    chunk_overlap = excluded.chunk_overlap,
                    status = 'processing',
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    normalized_path,
                    content_hash,
                    document_id,
                    chunk_size,
                    chunk_overlap,
                ),
            )

    def mark_indexed(self, source_path):
        self._update_status(source_path, "indexed", None)

    def mark_failed(self, source_path, error_message):
        self._update_status(source_path, "failed", str(error_message))

    def _update_status(self, source_path, status, error_message):
        normalized_path = str(Path(source_path).resolve())

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE ingestion_manifest
                SET status = ?,
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE source_path = ?
                """,
                (status, error_message, normalized_path),
            )

            if cursor.rowcount == 0:
                raise KeyError(
                    f"No ingestion record found for {normalized_path}"
                )

    def list_all(self) -> list[dict]:
        """Return all tracked source files."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_manifest ORDER BY source_path"
            ).fetchall()

        return [dict(row) for row in rows]


    def remove_source(self, source_path: str) -> None:
        """Remove a source from the manifest after it is removed from the index."""
        normalized_path = str(Path(source_path).resolve())

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM ingestion_manifest WHERE source_path = ?",
                (normalized_path,),
            )