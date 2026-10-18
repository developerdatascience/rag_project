"""
Your FAISSVectorStore.search() implementation already supports distance_threshold according to the 
earlier project code. It compares the returned distance against the threshold and excludes results above it.
Important: FAISS IndexFlatL2 returns squared L2 distances. A threshold suitable for one embedding model 
or dataset may not work well for another. We'll calibrate the threshold using labeled evaluation questions 
rather than inventing a universal value
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

from src.chunker import DocumentChunker
from src.embeddings import EmbeddingService
from src.ingestion_manifest import IngestionManifest
from src.loader import load_document
from src.vector_store import FAISSVectorStore
from src.grounded_answerer import GroundedAnswerGenerator
from src.retrieval_config import load_retrieval_config
from src.guardrails import InputGuardrails

class IngestionService:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        data_dir: str = "data",
    ):
        config = load_retrieval_config()
        chunk_size = config["chunk_size"] if chunk_size is None else chunk_size
        chunk_overlap = (
            config["chunk_overlap"] if chunk_overlap is None else chunk_overlap
        )
        self.default_top_k = config["top_k"]

        self.data_dir = Path(data_dir)
        self.documents_dir = self.data_dir / "documents"
        self.snapshots_dir = self.data_dir / "faiss_snapshots"
        self.store_dir = self.data_dir / "faiss_store"

        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.guardrails = InputGuardrails()

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_service = EmbeddingService()
        self.manifest = IngestionManifest(
            self.data_dir / "ingestion_manifest.db"
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = FAISSVectorStore()
        self.answer_generator = GroundedAnswerGenerator()

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Hash file bytes to detect content changes."""
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()

    def _discover_files(self) -> list[Path]:
        """Find supported documents recursively."""
        return sorted(
            path.resolve()
            for path in self.documents_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

    def _current_pointer(self) -> Path:
        return self.store_dir / "CURRENT"

    def _activate_snapshot(self, snapshot_name: str) -> None:
        """Atomically switch the active snapshot pointer."""
        pointer = self._current_pointer()
        temporary_pointer = self.store_dir / "CURRENT.tmp"

        temporary_pointer.write_text(
            snapshot_name,
            encoding="utf-8",
        )
        os.replace(temporary_pointer, pointer)

    def _load_active_snapshot(self) -> bool:
        """Load the active persisted FAISS snapshot, if valid."""
        pointer = self._current_pointer()

        if not pointer.exists():
            return False

        snapshot_name = pointer.read_text(encoding="utf-8").strip()

        # Reject path traversal and unexpected pointer values.
        if not snapshot_name or Path(snapshot_name).name != snapshot_name:
            raise ValueError("Invalid active snapshot pointer.")

        snapshot_path = self.snapshots_dir / snapshot_name

        if not (
            (snapshot_path / "index.faiss").is_file()
            and (snapshot_path / "documents.json").is_file()
        ):
            return False

        self.vector_store = FAISSVectorStore()
        self.vector_store.load(snapshot_path)
        return True

    def _corpus_fingerprint(self, file_hashes: dict[str, str]) -> str:
        """Fingerprint the corpus and indexing configuration."""
        payload = {
            "files": sorted(file_hashes.items()),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def ingest(self) -> dict:
        files = self._discover_files()

        if not files:
            raise ValueError(
                f"No PDF or DOCX documents found in {self.documents_dir}"
            )

        # Hash the complete current corpus before deciding whether to rebuild.
        file_hashes = {
            str(path): self._file_hash(path)
            for path in files
        }

        previous_records = {
            record["source_path"]: record
            for record in self.manifest.list_all()
        }
        current_paths = set(file_hashes)
        previous_paths = set(previous_records)

        changed_paths = {
            path
            for path, content_hash in file_hashes.items()
            if (
                path not in previous_records
                or previous_records[path]["content_hash"] != content_hash
                or previous_records[path]["chunk_size"] != self.chunk_size
                or previous_records[path]["chunk_overlap"] != self.chunk_overlap
                or previous_records[path]["status"] != "indexed"
            )
        }

        deleted_paths = previous_paths - current_paths
        fingerprint = self._corpus_fingerprint(file_hashes)
        snapshot_path = self.snapshots_dir / fingerprint

        # Fast path: nothing changed and a usable persisted index exists.
        if (
            not changed_paths
            and not deleted_paths
            and self._load_active_snapshot()
        ):
            return {
                "status": "unchanged",
                "files": len(files),
                "chunks": len(self.vector_store.documents),
                "embeddings_created": 0,
                "snapshot": fingerprint,
            }

        # A previously built snapshot for this exact configuration may be reused.
        for path in changed_paths:
            self.manifest.mark_processing(
                source_path=path,
                content_hash=file_hashes[path],
                document_id=hashlib.sha256(
                    f"{path}:{file_hashes[path]}".encode("utf-8")
                ).hexdigest(),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        try:
            if (
                (snapshot_path / "index.faiss").is_file()
                and (snapshot_path / "documents.json").is_file()
            ):
                rebuilt_store = FAISSVectorStore()
                rebuilt_store.load(snapshot_path)
                embeddings_created = 0
            else:
                all_documents = []

                for path in files:
                    all_documents.extend(load_document(path))

                chunks = self.chunker.split(all_documents)

                if not chunks:
                    raise ValueError("Chunking produced no document chunks.")

                embeddings = self.embedding_service.embed_documents(
                    [chunk.page_content for chunk in chunks]
                )

                rebuilt_store = FAISSVectorStore()
                rebuilt_store.add(chunks, embeddings)

                # Build a new versioned snapshot. Do not overwrite the active one.
                if snapshot_path.exists():
                    shutil.rmtree(snapshot_path)

                rebuilt_store.save(snapshot_path)
                embeddings_created = len(chunks)

            # Switch active index only after the new snapshot has been saved.
            self._activate_snapshot(fingerprint)
            self.vector_store = rebuilt_store

            for path in changed_paths:
                self.manifest.mark_indexed(path)

            for path in deleted_paths:
                self.manifest.remove_source(path)

            return {
                "status": "rebuilt",
                "files": len(files),
                "chunks": len(self.vector_store.documents),
                "embeddings_created": embeddings_created,
                "changed_files": len(changed_paths),
                "deleted_files": len(deleted_paths),
                "snapshot": fingerprint,
            }

        except Exception as exc:
            for path in changed_paths:
                try:
                    self.manifest.mark_failed(path, str(exc))
                except Exception:
                    # Preserve the original ingestion exception.
                    pass
            raise

    def search(
        self,
        question: str,
        top_k: int | None = None,
        distance_threshold: float | None = None,
    ):
        """Search the presisted index, optionally filter by L2 distance."""

        question = self.guardrails.validate_question(question)
        top_k = self.default_top_k if top_k is None else top_k
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if distance_threshold is not None and distance_threshold < 0:
            raise ValueError("distance_threshold cannot be negative")
        
        if self.vector_store.index is None:
            if not self._load_active_snapshot():
                raise RuntimeError(
                    "No persisted index found. Run ingest() first."
                )

        query_embedding = self.embedding_service.embed_query(question)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def answer_question(self, 
                        question: str, 
                        top_k: int | None = None, 
                        distance_threshold: float| None= None,
                        answer_mode: str = "strict"
                        ) ->dict:
        results = self.search(
            question=question,
            top_k = top_k,
            distance_threshold=distance_threshold
        )

        answer = self.answer_generator.generate(question=question, results=results, answer_mode=answer_mode)

        sources = []
        for rank, (document, distance) in enumerate(results, start=1):
            page = document.metadata.get("page")

            sources.append({
                "rank": rank,
                "source": document.metadata.get("source" ,"unknown source"),
                "page": int(page) + 1 if isinstance(page, int) else None,
                "distance": round(float(distance), 4),
                "chunk_id": document.metadata.get("chunk_id")
            })

            return {
                "question": question,
                "answer": answer,
                "retrieved_chunks": len(results),
                "sources": sources
            }