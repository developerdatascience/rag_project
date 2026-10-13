from src.loader import PDFLoader
from src.chunker import DocumentChunker

PDF_PATH = "data/documents/understanding-the-dom.pdf"


def main():
    loader = PDFLoader()
    chunker = DocumentChunker(
        chunk_size=800,
        chunk_overlap=100,
    )

    documents = loader.load(PDF_PATH)
    chunks = chunker.split(documents)

    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 80)
        print(f"Chunk ID: {chunk.metadata.get('chunk_id')}")
        print(f"Document ID: {chunk.metadata.get('document_id')}")
        print(f"Chunk index: {chunk.metadata.get('chunk_index')}")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Page: {chunk.metadata.get('page')}")
        print(f"Characters: {len(chunk.page_content)}")
        print(chunk.page_content[:500])


if __name__ == "__main__":
    main()
