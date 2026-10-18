
import csv

import pytest

from evaluation.select_best_config import (
    load_candidates,
    select_best,
    write_json_atomically,
)


def test_selects_highest_f1():
    candidates = [
        {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k": 5,
            "f1": 0.38,
            "hit_rate": 0.75,
            "precision": 0.40,
            "recall": 0.37,
        },
        {
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "top_k": 3,
            "f1": 0.49,
            "hit_rate": 0.75,
            "precision": 0.50,
            "recall": 0.48,
        },
    ]

    best = select_best(candidates)

    assert best["chunk_size"] == 1200
    assert best["top_k"] == 3


def test_hit_rate_breaks_f1_tie():
    candidates = [
        {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k": 3,
            "f1": 0.40,
            "hit_rate": 0.50,
            "precision": 0.40,
            "recall": 0.40,
        },
        {
            "chunk_size": 700,
            "chunk_overlap": 100,
            "top_k": 5,
            "f1": 0.40,
            "hit_rate": 0.90,
            "precision": 0.40,
            "recall": 0.40,
        },
    ]

    assert select_best(candidates)["top_k"] == 5


def test_invalid_overlap_row_is_skipped(tmp_path):
    csv_path = tmp_path / "results.csv"

    fields = [
        "chunk_size", "chunk_overlap", "top_k",
        "recall", "precision", "f1", "hit_rate",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "chunk_size": 500,
            "chunk_overlap": 500,
            "top_k": 3,
            "recall": 0.4,
            "precision": 0.4,
            "f1": 0.4,
            "hit_rate": 0.5,
        })

    with pytest.raises(ValueError, match="No valid candidates"):
        load_candidates(csv_path)


def test_atomic_json_write(tmp_path):
    output = tmp_path / "config.json"
    write_json_atomically(output, {"top_k": 5})

    assert output.exists()
    assert '"top_k": 5' in output.read_text(encoding="utf-8")
