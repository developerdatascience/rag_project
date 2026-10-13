

def calculate_recall_at_k(retrieved_chunks, relevant_chunk_ids):
    """
    Of all known relevant chunks, what fraction was retrieved?

    Recall@K = relevant retrieved chunks / all relevant chunks
    """
    relevant_ids = set(relevant_chunk_ids)

    if not relevant_ids:
        return 0.0

    retrieved_ids = {
        doc.metadata["chunk_id"]
        for doc, _ in retrieved_chunks
    }

    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def calculate_precision_at_k(retrieved_chunks, relevant_chunk_ids):
    """
    What fraction of retrieved chunks is considered relevant?
    """
    if not retrieved_chunks:
        return 0.0

    relevant_ids = set(relevant_chunk_ids)
    relevant_retrieved = sum(
        doc.metadata["chunk_id"] in relevant_ids
        for doc, _ in retrieved_chunks
    )

    return relevant_retrieved / len(retrieved_chunks)


def calculate_page_hit_at_k(retrieved_chunks, relevant_pages):
    """
    Return 1 if at least one retrieved chunk comes from a
    relevant page; otherwise return 0.

    This metric assumes page numbers use PyPDFLoader's
    zero-based indexing.
    """

    relevant_pages = {int(page) for page in relevant_pages}
    if not relevant_pages:
        raise ValueError(f"Page hit rate requires at least one relevant page")

    retrieved_pages = {
        int(doc.metadata['page']) for doc, _ in retrieved_chunks
        if doc.metadata.get("page") is not None
    }

    return float(bool(retrieved_pages & relevant_pages))


def calculate_f1(precision, recall):
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def get_relevant_chunk_ids(chunks, relevant_pages):
    """
    Map page-level labels to the chunk IDs produced by
    the current chunking configuration.
    """
    relevant_pages = set(relevant_pages)

    return {
        chunk.metadata["chunk_id"]
        for chunk in chunks
        if chunk.metadata.get("page") in relevant_pages
    }

