
import argparse
import csv
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "recall",
    "precision",
    "f1",
    "hit_rate",
}


def parse_number(row, field):
    value = float(row[field])

    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")

    return value


def load_candidates(csv_path):
    candidates = []

    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV has no header row")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                chunk_size = int(row["chunk_size"])
                chunk_overlap = int(row["chunk_overlap"])
                top_k = int(row["top_k"])

                recall = parse_number(row, "recall")
                precision = parse_number(row, "precision")
                f1 = parse_number(row, "f1")
                hit_rate = parse_number(row, "hit_rate")

                if chunk_size <= 0 or top_k <= 0:
                    raise ValueError(
                        "chunk_size and top_k must be positive"
                    )

                if not 0 <= chunk_overlap < chunk_size:
                    raise ValueError(
                        "chunk_overlap must be >= 0 and < chunk_size"
                    )

                for name, value in (
                    ("recall", recall),
                    ("precision", precision),
                    ("f1", f1),
                    ("hit_rate", hit_rate),
                ):
                    if not 0 <= value <= 1:
                        raise ValueError(
                            f"{name} must be between 0 and 1"
                        )

                candidates.append({
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "top_k": top_k,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "hit_rate": hit_rate,
                    "csv_row": row_number,
                })

            except (ValueError, TypeError, KeyError) as exc:
                print(f"Skipping CSV row {row_number}: {exc}")

    if not candidates:
        raise ValueError(
            "No valid candidates found. Existing config was not changed."
        )

    return candidates


def select_best(candidates):
    # Primary: F1. Secondary: hit rate.
    # Final tie-breakers make selection deterministic.
    return max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["hit_rate"],
            item["precision"],
            item["recall"],
        ),
    )


def write_json_atomically(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")

        os.replace(temp_path, output_path)

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation_results.csv",
        help="CSV produced by your retrieval experiments",
    )
    parser.add_argument(
        "--output",
        default="data/retrieval_config.json",
        help="Where to write the selected configuration",
    )
    args = parser.parse_args()

    candidates = load_candidates(args.input)
    best = select_best(candidates)

    payload = {
        "chunk_size": best["chunk_size"],
        "chunk_overlap": best["chunk_overlap"],
        "top_k": best["top_k"],
        "selection_metric": "f1",
        "selection_score": best["f1"],
        "hit_rate": best["hit_rate"],
        "precision": best["precision"],
        "recall": best["recall"],
        "candidate_count": len(candidates),
        "selected_csv_row": best["csv_row"],
        "source_csv": str(Path(args.input)),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
        "notes": (
            "Selected from supplied experiment results. "
            "Validate on a separate holdout dataset before production use."
        ),
    }

    write_json_atomically(args.output, payload)

    print("Selected retrieval configuration:")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
