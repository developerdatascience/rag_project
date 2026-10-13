from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    total_chunks: int
    recall: float
    precision: float
    f1: float
    hit_rate: float