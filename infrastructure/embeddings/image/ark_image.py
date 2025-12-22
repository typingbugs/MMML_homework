from infrastructure.embeddings.image.base import ImageEmbeddingModel
import os
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    raise ImportError("Please install volcengine-python-sdk with 'pip install volcengine-python-sdk[ark]' to use Ark API.")


class ArkImageEmbedding(ImageEmbeddingModel):
    def __init__(self, base_url: str, model: str):
        self.client = Ark(api_key=os.environ.get("ARK_API_KEY"))
        self.model = model

    def embed_image(self, image_base64, image_type: str, text: str = None):
        assert image_type in ["png", "jpeg", "webp", "bmp"], "Unsupported image type"

        input_content = []
        if text:
            input_content.append({
                "type": "text",
                "text": text
            })
        input_content.append({
            "type": "image_url",
            "image_url": f"data:image/{image_type};base64,'{image_base64}'"
        })

        resp = self.client.multimodal_embeddings.create(
            model=self.model,
            input=input_content
        )

        return resp["data"]["embedding"]