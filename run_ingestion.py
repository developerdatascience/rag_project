
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

    top_k = config["top_k"]

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        results = service.search(question, top_k=top_k)

        if not results:
            print("No relevant chunks found.")
            continue

        for rank, (document, distance) in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print(f"Source: {document.metadata.get('source', 'Unknown')}")
            print(f"Page: {document.metadata.get('page', 'Unknown')}")
            print(f"L2 distance: {distance:.4f}")
            print(document.page_content[:1000])


if __name__ == "__main__":
    main()
