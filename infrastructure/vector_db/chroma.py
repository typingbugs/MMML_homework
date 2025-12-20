import chromadb
from .base import VectorDB
from typing import Literal

class ChromaDB(VectorDB):
    def __init__(self, persist_dir, collection_name: Literal["papers", "images"] = "papers"):
        self.client = chromadb.Client(
            settings=chromadb.Settings(
                persist_directory=persist_dir
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(self, texts, embeddings, metadatas):
        ids = [str(i) for i in range(len(texts))]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding, top_k=5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
