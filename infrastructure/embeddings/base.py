from abc import ABC, abstractmethod
from typing import List

class EmbeddingModel(ABC):

    @abstractmethod
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_image(self, image_path: str) -> List[float]:
        pass