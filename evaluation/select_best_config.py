
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
