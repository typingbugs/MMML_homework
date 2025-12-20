from abc import ABC, abstractmethod

class VectorDB(ABC):

    @abstractmethod
    def add(self, texts, embeddings, metadatas):
        pass

    @abstractmethod
    def search(self, query_embedding, top_k=5):
        pass
