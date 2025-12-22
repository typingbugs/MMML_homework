import os
from typing import List
from infrastructure.embeddings.text.base import TextEmbeddingModel
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    raise ImportError("Please install volcengine-python-sdk with 'pip install volcengine-python-sdk[ark]' to use Ark API.")


class ArkTextEmbedding(TextEmbeddingModel):
    def __init__(self, base_url: str, model: str):
        self.client = Ark(api_key=os.environ.get("ARK_API_KEY"))
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        input_content = [
            {
                "type": "text",
                "text": text
            }
        ]
        resp = self.client.multimodal_embeddings.create(
            model=self.model,
            input=input_content,
            encoding_format='float'
        )

        return resp.data.embedding