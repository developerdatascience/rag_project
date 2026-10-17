
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.qa_dataset import EVALUATION_CASES
from src.ingestion_service import IngestionService

from evaluation.answer_metrics import evaluate_answer


OUTPUT_DIR = Path("evaluation/reports")
VALID_ANSWERABILITY = {"supported", "unsupported", "pending_review"}


def extract_retrieved_text(results):
    """Combine retrieved document text for evidence matching."""
    return "\n".join(
        document.page_content
        for document, _distance in results
    )


def build_source_records(results):
    """Keep source details for later manual inspection."""
    sources = []

    for document, distance in results:
        metadata = document.metadata or {}
        page = metadata.get("page")

        # PyPDFLoader normally stores a zero-based page index.
        # Convert it to a one-based page number for display.
        if isinstance(page, int):
            page = page + 1

        sources.append(
            {
                "source": metadata.get("source", "unknown"),
                "page": page,
                "distance": float(distance),
                "chunk_id": metadata.get("chunk_id"),
                "content": document.page_content,
            }
        )

    return sources


def evaluate_case(service, case, top_k=3):
    case_id = case["id"]
    question = case["question"]
    answerability = case["answerability"]
    required_evidence = case["required_evidence"]

    if answerability not in VALID_ANSWERABILITY:
        raise ValueError(
            f"{case_id}: invalid answerability={answerability!r}"
        )

    if answerability == "supported" and not required_evidence:
        raise ValueError(
            f"{case_id}: supported cases need required_evidence"
        )

    if answerability == "unsupported" and required_evidence:
        raise ValueError(
            f"{case_id}: unsupported cases should have no "
            "positive required_evidence phrases"
        )

    results = service.search(question, top_k=top_k)
    retrieved_text = extract_retrieved_text(results)
    sources = build_source_records(results)

    matched_evidence = [
        phrase
        for phrase in required_evidence
        if phrase.casefold() in retrieved_text.casefold()
    ]

    if answerability == "pending_review":
        evaluation_status = "NEEDS_ANNOTATION"
        evidence_coverage = None

    elif answerability == "supported":
        evidence_coverage = (
            len(matched_evidence) / len(required_evidence)
        )
        evaluation_status = (
            "PASS_EVIDENCE"
            if evidence_coverage == 1.0
            else "FAIL_EVIDENCE"
        )

    else:
        # Absence of annotated evidence in top-k is not proof that
        # the document is unsupported. That label requires review.
        evidence_coverage = None
        evaluation_status = "REQUIRES_ABSTENTION_CHECK"

    # Generate an answer for inspection, using strict grounding.
    # This does not by itself prove the answer is correct.
    answer_result = service.answer_question(
        question,
        top_k=top_k,
        answer_mode="strict",
    )

    if isinstance(answer_result, dict):
        answer = answer_result.get("answer", str(answer_result))
    else:
        answer = str(answer_result)

    answer_metrics = evaluate_answer(
        answer=answer,
        answerability=answerability,
    )

    return {
        "id": case_id,
        "question": question,
        "answerability": answerability,
        "evaluation_status": evaluation_status,
        "top_k": top_k,
        "required_evidence": required_evidence,
        "matched_evidence": matched_evidence,
        "evidence_coverage": evidence_coverage,
        "answer": answer,
        "answer_metrics": answer_metrics,
        "sources": sources,
    }



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Uses the default constructor from the current project.
    # If your IngestionService constructor requires arguments,
    # pass the same configuration used by your working runner.
    service = IngestionService()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_k": 3,
        "cases": [],
    }

    for case in EVALUATION_CASES:
        result = evaluate_case(service, case, top_k=3)
        report["cases"].append(result)

        print("\n" + "=" * 80)
        print(f"CASE: {result['id']}")
        print(f"Status: {result['evaluation_status']}")
        print(f"Answerability: {result['answerability']}")

        if result["evidence_coverage"] is not None:
            print(
                "Evidence coverage: "
                f"{result['evidence_coverage']:.2%}"
            )
            print("Matched evidence:", result["matched_evidence"])

        print("\nGenerated answer:")
        print(result["answer"])

        print("\nAnswer metrics:")
        print(
            "Citation present:",
            result["answer_metrics"]["citation_present"],
        )
        print(
            "Abstention detected:",
            result["answer_metrics"]["abstention_detected"],
        )
        print(
            "Abstention correct:",
            result["answer_metrics"]["abstention_correct"],
        )
        print(
            "Manual answer review required:",
            result["answer_metrics"]["answer_review_required"],
        )

        print("\nRetrieved sources:")
        for index, source in enumerate(result["sources"], start=1):
            print(
                f"{index}. {source['source']} | "
                f"page={source['page']} | "
                f"distance={source['distance']:.4f}"
            )
            print(source["content"][:350].replace("\n", " "))

    report_path = OUTPUT_DIR / "latest_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
