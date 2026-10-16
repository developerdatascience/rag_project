# End to end - Complete RAG Project

## Project structure

```text
rag-project/
├── data/
│   └── documents/
│       └── understanding-the-dom.pdf
├── evaluation/
│   ├── config.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── results.py
│   └── retrieval_dataset.py
├── notebooks/
│   ├── experiment_1.ipynb
│   ├── experiment_2.ipynb
│   └── loader.ipynb
├── src/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── ingestion_manifest.py
│   ├── ingestion_service.py
│   ├── loader.py
│   ├── prompt.py
│   ├── rag.py
│   ├── retriever.py
│   └── vector_store.py
├── tests/
│   ├── test_chunker.py
│   ├── test_ingestion_manifest.py
│   ├── test_loader.py
│   └── test_metrics.py
├── .gitignore
├── LICENSE
├── README.md
├── evaluate_retrieval.py
├── inspect_chunks.py
├── inspect_retrieval.py
├── main.py
├── pytest.ini
├── requirements.txt
├── run_experiment.py
└── run_ingestion.py
```


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

**1. First run:** load documents, create chunks and embeddings, save the FAISS index, then mark ingestion as complete.
**2. Restart with unchanged documents:** load the saved index and skip unnecessary embedding work.
**3. Changed or added document:** rebuild the index from the current document collection, then update the manifest.
**4. Failed ingestion:** record the failure and leave the system able to retry.

#### 1. Update the manifest

Add these methods to *src/ingestion_manifest.py*. They let the ingestion service detect deleted files and remove their manifest entries after a successful rebuild.



#### 2. Create *src/ingestion_service.py*

This class coordinates loading, hashing, chunking, embedding, persistence, and manifest updates. It assumes the loader, chunker, embedding service, and persistent FAISSVectorStore from the previous milestones are available.

**Important:** this version uses a versioned snapshot directory and an atomically replaced CURRENT pointer, so the previous active snapshot remains available if building a new one fails before activation. It does not yet implement concurrent-writer locking or automatic cleanup of old snapshots; those are production hardening tasks.

**Scenario A — First ingestion**
Expected behavior:
- PDF and DOCX files are discovered.
- Documents are chunked and embedded.
- A versioned FAISS snapshot is saved.
- The CURRENT pointer is updated.
- Manifest records are marked indexed.

The output should report status: rebuilt.

**Scenario B — Restart without changing files**
Stop the process and run the script again.
Expected behavior:
- The existing snapshot is loaded.
- No new document embeddings are generated.
- The output reports status: unchanged.
The important distinction is that **query embeddings are still generated for searches.** Skipping ingestion embeddings does not mean skipping query embeddings.

**Scenario C — Modify or add a document**
Edit a document under data/documents/, save it, and run the script again.

Expected behavior:
- The changed file is detected through its content hash.
- The complete current corpus is re-indexed into a new snapshot.
- The new snapshot becomes active only after it has been saved.
- Deleted files are removed from the manifest after the rebuild succeeds.

If any scenario behaves differently, inspect the manifest at data/ingestion_manifest.db and the snapshots under data/faiss_snapshots/.

**6. Two important limitations to address next**

**1. Concurrent ingestion:** two processes could attempt to update the active snapshot simultaneously. Add a locking mechanism before running ingestion concurrently in production.
**2. Recovery and consistency:** if activation succeeds but the manifest update fails, the next run may rebuild unnecessarily. This is safe for the current approach, but a transactional ingestion journal and recovery procedure would make the workflow more robust.

Also, FAISS uses squared L2 distance in this implementation, so a smaller distance means a closer vector. Do not interpret the distance directly as a probability or percentage of relevance.

### 2. Next milestone — adaptive retrieval
Now we’ll move from **storing and reusing vectors** to choosing retrieval settings based on measured quality.
One important caveat: your latest index contains only one file and three chunks. Your earlier DOM-document evaluation results should not be used to tune this new corpus unless you restore that document and rerun the experiments. A three-chunk document is useful for testing persistence, but too small to establish reliable retrieval settings.
Our tuning process will compare:
- chunk_size: how much text each chunk contains.
- chunk_overlap: how much text neighboring chunks share.
- top_k: how many chunks are retrieved for each question.
- F1 and hit_rate: measures used to compare retrieval quality.
We’ll use a measured score rather than automatically choosing the largest top_k. Returning more chunks can improve recall while also introducing irrelevant context and increasing LLM costs.

### 3. Add an evaluation-driven configuration selector
Create `evaluation/select_best_config.py`. It reads your experiment results, ranks the candidate configurations, and saves the winning settings to `data/retrieval_config.json`.

For the initial version, the selector will use this transparent scoring formula:

> **Score = 0.7 × F1 + 0.3 × hit rate**

This is a tunable engineering choice, not a universal RAG standard. It balances retrieval precision and recall with the chance of retrieving at least one relevant chunk.

```python
import csv
import json
import math
from pathlib import Path


RESULTS_PATH = Path("evaluation_results.csv")
CONFIG_PATH = Path("data/retrieval_config.json")


def as_float(row: dict, field: str) -> float:
    value = float(row[field])

    if not math.isfinite(value):
        raise ValueError(f"Invalid {field} value: {value}")

    return value


def select_best_config(results_path: Path = RESULTS_PATH) -> dict:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find {results_path}. Run the retrieval experiments first."
        )

    with results_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The evaluation results file is empty.")

    candidates = []

    for row in rows:
        f1 = as_float(row, "f1")
        hit_rate = as_float(row, "hit_rate")

        if not 0 <= f1 <= 1 or not 0 <= hit_rate <= 1:
            raise ValueError("F1 and hit_rate must be between 0 and 1.")

        candidate = {
            "chunk_size": int(row["chunk_size"]),
            "chunk_overlap": int(row["chunk_overlap"]),
            "top_k": int(row["top_k"]),
            "f1": f1,
            "hit_rate": hit_rate,
            "precision": as_float(row, "precision"),
            "recall": as_float(row, "recall"),
            "total_chunks": int(row["total_chunks"]),
        }

        if candidate["chunk_size"] <= 0:
            raise ValueError("chunk_size must be positive.")

        if not 0 <= candidate["chunk_overlap"] < candidate["chunk_size"]:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        if candidate["top_k"] <= 0:
            raise ValueError("top_k must be positive.")

        candidate["score"] = 0.7 * f1 + 0.3 * hit_rate
        candidates.append(candidate)

    # Deterministic tie-breaks: higher F1, then lower top_k.
    candidates.sort(
        key=lambda item: (
            item["score"],
            item["f1"],
            -item["top_k"],
        ),
        reverse=True,
    )

    best = candidates[0]

    config = {
        "chunk_size": best["chunk_size"],
        "chunk_overlap": best["chunk_overlap"],
        "top_k": best["top_k"],
        "selection_score": round(best["score"], 4),
        "evaluation_metrics": {
            "f1": best["f1"],
            "hit_rate": best["hit_rate"],
            "precision": best["precision"],
            "recall": best["recall"],
        },
        "evaluation_results_file": str(results_path),
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    return config


if __name__ == "__main__":
    selected = select_best_config()

    print("Selected retrieval configuration")
    print("-" * 40)

    for key, value in selected.items():
        print(f"{key}: {value}")

    print(f"\nSaved configuration to: {CONFIG_PATH}")
```

Run the selector from the project root with:

```bash
python evaluation/select_best_config.py
```



***4. Connect the selected configuration to ingestion**
Update *run_ingestion.py* so it reads the selected settings rather than hardcoding 1200/150.

