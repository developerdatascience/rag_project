
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.qa_dataset import EVALUATION_CASES
from src.ingestion_service import IngestionService


TOP_K_VALUES = [3, 5, 10]
OUTPUT_DIR = Path("evaluation/reports")


def evaluate_retrieval(service, case, top_k):
    """Measure exact evidence-phrase coverage at a given top_k."""
    results = service.search(
        case["question"],
        top_k=top_k,
    )

    retrieved_text = "\n".join(
        document.page_content
        for document, _distance in results
    ).casefold()

    required_evidence = case.get("required_evidence", [])

    matched = [
        phrase
        for phrase in required_evidence
        if phrase.casefold() in retrieved_text
    ]

    coverage = (
        len(matched) / len(required_evidence)
        if required_evidence
        else None
    )

    sources = []

    for document, distance in results:
        metadata = document.metadata or {}
        page = metadata.get("page")

        if isinstance(page, int):
            page += 1

        sources.append({
            "source": metadata.get("source", "unknown"),
            "page": page,
            "distance": float(distance),
            "chunk_id": metadata.get("chunk_id"),
        })

    return {
        "id": case["id"],
        "question": case["question"],
        "answerability": case["answerability"],
        "top_k": top_k,
        "required_evidence_count": len(required_evidence),
        "matched_evidence": matched,
        "evidence_coverage": coverage,
        "sources": sources,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Match the constructor to your existing working runner if needed.
    service = IngestionService()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_k_values": TOP_K_VALUES,
        "cases": [],
    }

    for case in EVALUATION_CASES:
        if case["answerability"] != "supported":
            print(
                f"Skipping {case['id']}: "
                f"answerability={case['answerability']}"
            )
            continue

        if not case.get("required_evidence"):
            print(
                f"Skipping {case['id']}: "
                "no required evidence phrases"
            )
            continue

        for top_k in TOP_K_VALUES:
            result = evaluate_retrieval(service, case, top_k)
            report["cases"].append(result)

            coverage = result["evidence_coverage"]

            print(
                f"{result['id']} | top_k={top_k} | "
                f"evidence coverage={coverage:.1%}"
            )
            print("  Matched:", result["matched_evidence"])

            for source in result["sources"]:
                print(
                    f"  Source: {source['source']} | "
                    f"page={source['page']} | "
                    f"distance={source['distance']:.4f}"
                )

    output_path = OUTPUT_DIR / "retrieval_diagnostics.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
