from dataclasses import dataclass

@dataclass(frozen=True)
class RetrievalConfig:
    chunk_size: int
    chunk_overlap: int
    top_k: int

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap <= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be >=0 and smaller than chunk_size"
            )
        if self.top_k <=0:
            raise ValueError("top_kmust be positive")