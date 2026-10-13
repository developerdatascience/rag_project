import csv

from evaluation.experiment import RetrievalExperiment


CHUNKING_CONFIGS = [
    (500, 50),
    (800, 100),
    (1200, 150),
]

TOP_K_VALUES = [1, 3, 5, 10]


def main():
    experiment = RetrievalExperiment()
    all_results = []

    for chunk_size, chunk_overlap in CHUNKING_CONFIGS:
        print(
            f"\nTesting chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

        results = experiment.run_chunking_configuration(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k_values=TOP_K_VALUES,
        )
        all_results.extend(results)

        for result in results:
            print(
                f"K={result.top_k:2} | "
                f"chunks={result.total_chunks:4} | "
                f"Recall={result.recall:.4f} | "
                f"Precision={result.precision:.4f} | "
                f"F1={result.f1:.4f} |"
                f"PageHitRate={result.hit_rate:.4f}"
            )

    output_path = "evaluation_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "chunk_size",
                "chunk_overlap",
                "top_k",
                "total_chunks",
                "recall",
                "precision",
                "f1",
                "hit_rate"
            ],
        )
        writer.writeheader()

        for result in all_results:
            writer.writerow({
                "chunk_size": result.chunk_size,
                "chunk_overlap": result.chunk_overlap,
                "top_k": result.top_k,
                "total_chunks": result.total_chunks,
                "recall": round(result.recall, 4),
                "precision": round(result.precision, 4),
                "f1": round(result.f1, 4),
                "hit_rate": round(result.hit_rate, 2)
            })

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

