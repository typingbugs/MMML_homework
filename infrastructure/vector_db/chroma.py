import chromadb
from infrastructure.vector_db.base import VectorDB
from typing import Literal
import json
import uuid

class ChromaDB(VectorDB):
    def __init__(self, persist_dir, collection_name: Literal["papers", "images"] = "papers"):
        self.client = chromadb.PersistentClient(
            path=persist_dir
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(self, texts, embeddings, metadatas):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        for metadata in metadatas:
            for k, v in metadata.items():
                if isinstance(v, list):
                    metadata[k] = json.dumps(v)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding, top_k=5):
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        for metadata in res["metadatas"][0]:
            for k, v in metadata.items():
                try:
                    metadata[k] = json.loads(v)
                except:
                    pass

        return res
