from utils.pdf_loader import load_pdf_text, load_papers_from_dir
from pathlib import Path
from shutil import copyfile
import hashlib
from typing import List
from infrastructure.embeddings import OpenAITextEmbedding
from infrastructure.vector_db.chroma import ChromaDB


class PaperService:
    def __init__(self, embedder, vector_db, save_dir):
        self.embedder: OpenAITextEmbedding = embedder
        self.vector_db: ChromaDB = vector_db
        self.save_dir = Path(save_dir)

    def add_paper(self, pdf_path, topics=None):
        pdf_path = Path(pdf_path)

        if pdf_path.is_file():
            chunks, embeddings, metadatas = self.embed_paper_from_file(str(pdf_path), topics=topics)
        elif pdf_path.is_dir():
            chunks, embeddings, metadatas = self.embed_paper_from_dir(str(pdf_path), topics=topics)

        self.vector_db.add(
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"✅ 已索引论文: {pdf_path}")

    def embed_paper_from_file(self, pdf_path: str, topics=None):
        text = load_pdf_text(Path(pdf_path))
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        embeddings = self.make_request(chunks, topics=topics)
        save_paths = self.save_paper(pdf_path, topics=topics)
        metadatas = [{"path": save_paths}] * len(chunks)
        return chunks, embeddings, metadatas
    
    def embed_paper_from_dir(self, pdf_dir: str, topics=None):
        pdf_paths = load_papers_from_dir(pdf_dir)

        all_chunks = []
        all_embeddings = []
        all_metadatas = []

        for path in pdf_paths:
            chunks, embeddings, metadatas = self.embed_paper_from_file(path, topics=topics)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            all_metadatas.extend(metadatas)

        return all_chunks, all_embeddings, all_metadatas

    def search(self, query, top_k=5):
        q_emb = self.make_request([query])[0]
        results = self.vector_db.search(q_emb, top_k=top_k)

        if len(results["documents"][0]) == 0:
            print("🔍 未找到相关论文。")
            return

        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            print(f"\n--- Result {i+1} ---")
            print("来源文件:", meta["path"][0])
            print(doc[:300], "...")

    def save_paper(self, pdf_path: str, topics: str = None):
        topics = topics.split(",") if topics else ["untagged"]
        pdf_path = Path(pdf_path)
        save_file_name = hashlib.md5(str(pdf_path).encode()).hexdigest() + pdf_path.suffix
        save_paths = []
        for topic in topics:
            save_dir = self.save_dir / topic
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / save_file_name
            copyfile(pdf_path, save_path)
            save_paths.append(save_path)
        return save_paths
    
    def make_request(self, texts: List[str], topics=None):
        if topics:
            topics = f"Topics: {topics}" if topics else ""
            texts = [f"Passage: {text}" for text in texts]
            input_texts = [topics + "\n" + text for text in texts]
        else:
            input_texts = texts

        emb = [
            self.embedder.embed_text(input_text)
            for input_text in input_texts
        ]
        return emb