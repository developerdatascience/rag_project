
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "top_k": 3,
}


def load_retrieval_config(
    path="data/retrieval_config.json",
):
    config_path = Path(path)

    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    merged = {**DEFAULT_CONFIG, **config}

    chunk_size = merged["chunk_size"]
    chunk_overlap = merged["chunk_overlap"]
    top_k = merged["top_k"]

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if (
        not isinstance(chunk_overlap, int)
        or not 0 <= chunk_overlap < chunk_size
    ):
        raise ValueError(
            "chunk_overlap must be >= 0 and less than chunk_size"
        )

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }
