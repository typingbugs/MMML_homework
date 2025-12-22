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
        emb = self.make_image_request(image_path, topics=topics)
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
        if Path(query).is_file():
            q_emb = self.make_image_request(query)
        else:
            q_emb = self.make_text_request(query)

        results = self.vector_db.search(q_emb, top_k=top_k)

        if len(results["metadatas"][0]) == 0:
            print("🔍 未找到相关图片。")
            return

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
        
    def make_image_request(self, image_path: str, topics=None):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        emb = self.image_embedder.embed_image(
            image_base64, 
            image_type=Path(image_path).suffix.lstrip("."),
            text=f"Topics: {topics}\n" if topics else None
        )
        return emb
    
    def make_text_request(self, text: str):
        emb = self.text_embedder.embed_text(text)
        return emb