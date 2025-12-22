from abc import ABC, abstractmethod
from typing import List

class ImageEmbeddingModel(ABC):

    @abstractmethod
    def embed_image(self, image_path: str, topics: str = None) -> List[float]:
        pass
