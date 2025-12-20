from utils.image_loader import load_images_from_dir
from pathlib import Path
from shutil import copyfile
import hashlib
from infrastructure.embeddings.openai_multimodal import OpenAIMultimodalEmbedding
from infrastructure.vector_db.chroma import ChromaDB


class ImageService:
    def __init__(self, embedder, vector_db, save_dir):
        self.embedder: OpenAIMultimodalEmbedding = embedder
        self.vector_db: ChromaDB = vector_db
        self.save_dir = Path(save_dir)

    def add_image(self, image_path, topics=None):
        image_path = Path(image_path)
        if image_path.is_file():
            image_path, emb, metadata = self.embed_image_from_file(str(image_path), topics=topics)
            documents, embeddings, metadatas = [image_path], [emb], [metadata]
        elif image_path.is_dir():
            documents, embeddings, metadatas = self.embed_images_from_dir(str(image_path), topics=topics)
        
        self.vector_db.add(
            texts=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"✅ 已索引图片: {image_path}")

    def embed_image_from_file(self, image_path: str, topics=None):
        emb = self.embedder.embed_image(image_path, topics=topics)
        save_paths = self.save_image(image_path, topics=topics)
        return image_path, emb, {"path": save_paths}

    def embed_images_from_dir(self, image_dir: str, topics=None):
        image_paths = load_images_from_dir(image_dir)

        embeddings = []
        metadatas = []
        documents = []
        
        for path in image_paths:
            image_path, emb, metadata = self.embed_image_from_file(path, topics=topics)
            embeddings.append(emb)
            documents.append(image_path)
            metadatas.append(metadata)

        return documents, embeddings, metadatas

    def search(self, query, top_k=5):
        q_emb = self.embedder.embed_text([query])[0]

        results = self.vector_db.search(q_emb, top_k=top_k)

        print("🔍 搜索结果：")
        for i, meta in enumerate(results["metadatas"][0]):
            print(f"{i+1}. {meta['path'][0]}")

    def save_image(self, image_path: str, topics=None):
        topics = topics.split(",") if topics else ["untagged"]
        image_path = Path(image_path)
        save_file_name = hashlib.md5(str(image_path).encode()).hexdigest() + image_path.suffix
        save_paths = []
        for topic in topics:
            save_dir = self.save_dir / topic
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / save_file_name
            copyfile(image_path, save_path)
            save_paths.append(save_path)
        return save_paths
        