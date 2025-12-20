from openai import OpenAI
from .base import EmbeddingModel

class OpenAITextEmbedding(EmbeddingModel):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_text(self, texts, topics=None):
        topics = f"Topics: {topics}" if topics else ""
        texts = [f"Passage: {text}" for text in texts]
        input_texts = [topics + "\n" + text for text in texts]
        resp = self.client.embeddings.create(
            model=self.model,
            input=input_texts
        )
        return [d.embedding for d in resp.data]
