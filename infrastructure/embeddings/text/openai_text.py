from openai import OpenAI
from typing import List
import os
from infrastructure.embeddings.text.base import TextEmbeddingModel


class OpenAITextEmbedding(TextEmbeddingModel):
    def __init__(self, base_url: str, model: str):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"), 
            base_url=base_url
        )
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return resp.data[0].embedding
