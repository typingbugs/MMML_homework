import base64
from openai import OpenAI
from .base import EmbeddingModel

class OpenAIMultimodalEmbedding(EmbeddingModel):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_text(self, texts):
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [d.embedding for d in resp.data]

    def embed_image(self, image_path: str, topics=None):
        input_content = []
        if topics:
            input_content.append({
                "type": "text",
                "text": f"Topics: {topics}\n"
            })
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        input_content.append({
            "type": "input_image",
            "image_base64": image_base64
        })

        resp = self.client.embeddings.create(
            model=self.model,
            input=input_content
        )
        return resp.data[0].embedding
