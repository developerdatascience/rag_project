
import json
from pathlib import Path

from src.ingestion_service import IngestionService


def main():
    config_path = Path("data/retrieval_config.json")

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        # Safe fallback until evaluation produces a configuration.
        config = {
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "top_k": 3,
        }

    service = IngestionService(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )

    result = service.ingest()

    print("\nIngestion result")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")

    top_k = config.get("top_k", 3)
    distance_threshold = config.get("distance_threshold")

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            result = service.answer_question(
                question=question,
                top_k=top_k,
                distance_threshold=distance_threshold
            )

            print("\nAnswer")
            print("-"*40)
            print(result["answer"])

            print("\nRetrieved sources")
            for source in result["sources"]:
                print(
                    f"- {source['source']}, "
                    f"page={source['page']}, "
                    f"distance={source['distance']}"
                )
        except Exception as exc:
            print(f"Unable to answer this question: {exc}")


if __name__ == "__main__":
    main()
