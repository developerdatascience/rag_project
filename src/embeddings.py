from langchain_openai import OpenAIEmbeddings
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

class EmbeddingService:
    """Generate embeddings for documents and queries"""

    def __init__(self, 
                 model_name: str = os.environ.get("EMBEDDING_MODEL"), 
                 base_url: str = os.environ.get("BASE_URL"), 
                 api_key: str = os.environ.get("OPENAI_API_KEY")) -> None:
        self.model = OpenAIEmbeddings(model=model_name, base_url=base_url, api_key=api_key)

    def embed_documents(
            self, texts: list[str]
            ) -> List[List[float]]:
        if not texts:
            raise ValueError("Cannot generate embeddings for empty input")

        return self.model.embed_documents(texts)


    def embed_query(self, query: str) -> List[float]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return self.model.embed_query(query)