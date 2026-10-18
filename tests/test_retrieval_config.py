
import json
import pytest

from src.retrieval_config import load_retrieval_config


def test_defaults_when_config_file_is_missing(tmp_path):
    config = load_retrieval_config(tmp_path / "missing.json")

    assert config == {
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "top_k": 3,
    }


def test_loads_config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({
            "chunk_size": 700,
            "chunk_overlap": 100,
            "top_k": 5,
        }),
        encoding="utf-8",
    )

    assert load_retrieval_config(path) == {
        "chunk_size": 700,
        "chunk_overlap": 100,
        "top_k": 5,
    }


def test_rejects_invalid_chunk_overlap(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({
            "chunk_size": 500,
            "chunk_overlap": 500,
            "top_k": 3,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_retrieval_config(path)
