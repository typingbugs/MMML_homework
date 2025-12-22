from typing import List
from utils.image_loader import load_images_from_dir
from pathlib import Path
from shutil import copyfile
import hashlib
import base64
from infrastructure.embeddings import ArkImageEmbedding, ArkTextEmbedding
from infrastructure.vector_db.chroma import ChromaDB


class ImageService:
    def __init__(self, image_embedder, text_embedder, vector_db, save_dir):
        self.image_embedder: ArkImageEmbedding = image_embedder
        self.text_embedder: ArkTextEmbedding = text_embedder
        self.vector_db: ChromaDB = vector_db
        self.save_dir = Path(save_dir)

    def add_image(self, image_path, topics=None):
        image_path = Path(image_path)
        image_paths, embeddings, metadatas = (
            self.embed_image_from_file(str(image_path), topics=topics)
            if image_path.is_file()
            else self.embed_images_from_dir(str(image_path), topics=topics)
        )

        self.vector_db.add(texts=image_paths, embeddings=embeddings, metadatas=metadatas)
        print(f"✅ 已索引图片: {image_path}")

    def embed_image_from_file(self, image_path: str, topics: str = None):
        topics = topics.split(",") if topics else []
        emb = self.make_image_request(image_path, topics=topics)
        if len(topics) == 0:
            topics = self.get_topic(emb, search_top_k=3, num_return=1)
        save_paths = self.save_image(image_path, topics=topics)
        return [image_path], [emb], [{"path": save_paths, "topics": topics}]

    def get_topic(self, embedding, search_top_k: int = 3, num_return: int = 1) -> List[str]:
        topic_counts = {}
        search_results = self.search(embedding=embedding, top_k=search_top_k)
        for search_result in search_results:
            for topic in search_result["topics"]:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        top_topics = [topic for topic, count in sorted_topics[:num_return]]
        if len(top_topics) == 0:
            top_topics = ["unknown"]
        return top_topics

    def embed_images_from_dir(self, image_dir: str, topics=None):
        image_paths = load_images_from_dir(image_dir)

        embeddings, metadatas, images = [], [], []
        
        for path in image_paths:
            image_path, emb, metadata = self.embed_image_from_file(path, topics=topics)
            embeddings.extend(emb)
            images.extend(image_path)
            metadatas.extend(metadata)

        return images, embeddings, metadatas

    def search(self, query: str=None, embedding=None, top_k=5):
        assert query or embedding, "必须提供查询文本或嵌入向量。"
        q_emb = (
            embedding if embedding is not None 
            else (
                self.make_image_request(query) 
                if Path(query).is_file() 
                else self.make_text_request(query)
            )
        )

        results = self.vector_db.search(q_emb, top_k=top_k)

        if len(results["metadatas"][0]) == 0:
            return []

        res = []
        for i, meta in enumerate(results["metadatas"][0]):
            res.append({
                "path": meta["path"][0],
                "topics": meta["topics"]
            })
        return res
    
    def search_image(self, query, top_k=5):
        results = self.search(query=query, top_k=top_k)

        if len(results) == 0:
            print("🔍 未找到相关图片。")
            return

        for i, result in enumerate(results):
            print(f"--- 结果 {i + 1} ---")
            print(f"路径: {result['path']}")
            print(f"主题: {', '.join(result['topics'])}")
            print()

    def save_image(self, image_path: str, topics: List[str]):
        image_path = Path(image_path)
        save_file_name = hashlib.md5(str(image_path).encode()).hexdigest() + image_path.suffix
        save_paths = []
        for topic in topics:
            save_dir = self.save_dir / topic
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / save_file_name
            copyfile(image_path, save_path)
            save_paths.append(str(save_path))
        return save_paths
        
    def make_image_request(self, image_path: str, topics: List[str] = None):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        emb = self.image_embedder.embed_image(
            image_base64, 
            image_type=Path(image_path).suffix.lstrip("."),
            text=f"Topics: {','.join(topics)}" if topics else None
        )
        return emb
    
    def make_text_request(self, text: str):
        emb = self.text_embedder.embed_text(text)
        return emb