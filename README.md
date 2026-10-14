# rag_project
RAG project



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