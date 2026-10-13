from evaluation.retrieval_dataset import EVALUATION_DATASET

from src.loader import PDFLoader
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore
from src.retriever import DocumentRetriever

from evaluation.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_f1,
)


from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/documents/understanding-the-dom.pdf"


def build_retriever(chunk_size: int, chunk_overlap: int):
    loader = PDFLoader()
    documents = loader.load(PDF_PATH)

    chunker = DocumentChunker(chunk_size=chunk_size, 
                              chunk_overlap=chunk_overlap,
                            )
    chunks = chunker.split(documents)

    embedding_service = EmbeddingService()

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_service.embed_documents(
        [chunk.page_content for chunk in chunks
         ]
    )
    vector_store = FAISSVectorStore()

    vector_store.add(documents=chunks, embeddings=embeddings)
    retriever = DocumentRetriever(
        embedding_service=embedding_service, vector_store=vector_store
    )
    return retriever, chunks


def get_relevant_chunk_ids(chunks, relevant_pages):
    relevant_pages = set(relevant_pages)
    return {
        chunk.metadata["chunk_id"]
        for chunk in chunks
        if chunk.metadata.get("page") in relevant_pages
    }



def evaluate_question(
    retriever,
    question_data,
    top_k,
):
    question = question_data["question"]

    relevant_chunk_ids = question_data[
        "relevant_chunks"
    ]

    results = retriever.retrieve(
        question=question,
        top_k=top_k,
    )

    recall = calculate_recall_at_k(
        retrieved_chunks=results,
        relevant_chunk_ids=relevant_chunk_ids,
    )

    precision = calculate_precision_at_k(
        retrieved_chunks=results,
        relevant_chunk_ids=relevant_chunk_ids,
    )

    f1 = calculate_f1(
        precision=precision,
        recall=recall,
    )

    retrieved_ids = [
        document.metadata["chunk_id"]
        for document, _ in results
    ]

    return {
        "question": question,
        "top_k": top_k,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "retrieved_chunks": retrieved_ids,
        "relevant_chunks": relevant_chunk_ids,
    }


def run_experiment():
    configurations = [
        (500, 50),
        (800, 100),
        (1200, 150)
    ]

    k_values = [1, 3, 5, 10]
    for chunk_size, chunk_overlap in configurations:
        retriever, chunks = build_retriever(
            chunk_size = chunk_size, 
            chunk_overlap=chunk_overlap
        )

        for k in k_values:
            recalls = []
            precisions = []

            for item in EVALUATION_DATASET:
                relevant_chunk_ids = get_relevant_chunk_ids(
                    chunks, 
                    item["relevant_pages"]
                )

                results = retriever.retrieve(
                    question = item["question"],
                    top_k = k
                )

                recalls.append(
                    calculate_recall_at_k(
                        retrieved_chunks=results,
                        relevant_chunk_ids=relevant_chunk_ids
                    )
                )

                precisions.append(
                    calculate_precision_at_k(
                        retrieved_chunks=results,
                        relevant_chunk_ids=relevant_chunk_ids
                    )
                )

            avg_recall = sum(recalls) / len(recalls)
            avg_precision = (
                sum(precisions) / len(precisions)
            )

            f1 = calculate_f1(
                avg_precision,
                avg_recall,
            )

            print(
                f"K={k:2} | "
                f"Recall={avg_recall:.3f} | "
                f"Precision={avg_precision:.3f} | "
                f"F1={f1:.3f}"
            )


def main():

    retriever = build_retriever()

    k_values = [1, 3, 5, 10]

    for k in k_values:

        print("\n" + "=" * 70)

        print(
            f"RETRIEVAL EVALUATION @ K={k}"
        )

        print("=" * 70)

        recalls = []
        precisions = []

        for question_data in EVALUATION_DATASET:

            result = evaluate_question(
                retriever=retriever,
                question_data=question_data,
                top_k=k,
            )

            print(
                f"\nQuestion: "
                f"{result['question']}"
            )

            print(
                f"Relevant chunks: "
                f"{result['relevant_chunks']}"
            )

            print(
                f"Retrieved chunks: "
                f"{result['retrieved_chunks']}"
            )

            print(
                f"Recall@{k}: "
                f"{result['recall']:.2f}"
            )

            print(
                f"Precision@{k}: "
                f"{result['precision']:.2f}"
            )

            recalls.append(
                result["recall"]
            )

            precisions.append(
                result["precision"]
            )

        average_recall = (
            sum(recalls) / len(recalls)
        )

        average_precision = (
            sum(precisions)
            / len(precisions)
        )

        print("\n" + "-" * 70)

        print(
            f"Average Recall@{k}: "
            f"{average_recall:.2f}"
        )

        print(
            f"Average Precision@{k}: "
            f"{average_precision:.2f}"
        )




if __name__ == "__main__":
    main()



