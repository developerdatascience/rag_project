# End to end - Complete RAG Project


What the manifest guarantees
- It identifies whether the file contents or chunking configuration changed.
- It distinguishes an in-progress, successful, or failed ingestion.
- It records failures so they can be diagnosed and retried.
- It does not claim a document was indexed until the indexing operation reports success.

Step 4 — How we will connect this to FAISS
The safe ingestion flow should eventually look like this:
1. Calculate the source file's content hash.
2. Check the manifest for the current document version and chunking configuration.
3. If unchanged and its corresponding vector index is available, skip embedding.
4. Otherwise, load and chunk the document.
5. Generate embeddings and build or update the index.
6. Verify that the index operation succeeded.
7. Mark the manifest entry as indexed.

For changed documents, we must remove the old vectors or rebuild the index from the current corpus. Simply adding the new chunks will leave stale content searchable.

For now, do not connect needs_ingestion() directly to the existing in-memory FAISS index as a cross-restart optimization. The manifest persists across restarts, but the index does not necessarily persist with it. The next implementation milestone is to give the vector store a well-defined lifecycle and rebuild strategy.


### Step 5 — Make the vector store persistent
Before integrating the manifest into the ingestion pipeline, we need to save and reload the FAISS index. Otherwise, after restarting the application, the manifest might say a document is indexed even though the in-memory index has disappeared.

Update src/vector_store.py to add persistence while retaining your existing add() and search() interface.

**Important implementation notes**
1. **Full rebuild is our first safe replacement strategy.** When a document changes, rebuild from the current corpus rather than appending its new chunks to an index containing stale chunks.
2. save() and load() persist the vector index and document metadata, but the two files are not a fully transactional snapshot. Before production, we should store them as versioned snapshots and atomically switch a pointer to the active snapshot.
3. Use only trusted index files when loading FAISS indexes. The JSON metadata file should also be validated before it is used.
4. Your existing DocumentRetriever should continue to call search() as before. Run your tests after replacing the implementation to catch any differences in the current integration.
Once persistence and rebuild behavior are verified, we can connect IngestionManifest to this lifecycle and ensure a document is marked indexed only after the new index snapshot is saved successfully.


We’ll continue with the next milestone: **integrating persistent FAISS storage with the SQLite ingestion manifest.**
The key reliability rule is that the manifest must never report a document as indexed before its corresponding vector index has been saved successfully. For this first implementation, we’ll use a safe **full-rebuild strategy** whenever a document changes. This avoids duplicate or stale vectors; incremental updates can come later.

**Milestone: Persistent ingestion pipeline**

The pipeline will support these scenarios:

1. First run: load documents, create chunks and embeddings, save the FAISS index, then mark ingestion as complete.
2. Restart with unchanged documents: load the saved index and skip unnecessary embedding work.
3. Changed or added document: rebuild the index from the current document collection, then update the manifest.
4. Failed ingestion: record the failure and leave the system able to retry.

#### 1. Update the manifest

Add these methods to *src/ingestion_manifest.py*. They let the ingestion service detect deleted files and remove their manifest entries after a successful rebuild.

#### 2. Create *src/ingestion_service.py*

This class coordinates loading, hashing, chunking, embedding, persistence, and manifest updates. It assumes the loader, chunker, embedding service, and persistent FAISSVectorStore from the previous milestones are available.

**Important:** this version uses a versioned snapshot directory and an atomically replaced CURRENT pointer, so the previous active snapshot remains available if building a new one fails before activation. It does not yet implement concurrent-writer locking or automatic cleanup of old snapshots; those are production hardening tasks.