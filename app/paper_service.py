from utils.pdf_loader import load_pdf_text, load_papers_from_dir
from pathlib import Path
from shutil import copyfile
import hashlib
from typing import List
from tqdm import tqdm
from infrastructure.embeddings import OpenAITextEmbedding
from infrastructure.vector_db.chroma import ChromaDB


class PaperService:
    def __init__(self, embedder, vector_db, save_dir):
        self.embedder: OpenAITextEmbedding = embedder
        self.vector_db: ChromaDB = vector_db
        self.save_dir = Path(save_dir)

    def add_paper(self, pdf_path, topics=None):
        pdf_path = Path(pdf_path)

        chunks, embeddings, metadatas = (
            self.embed_paper_from_file(str(pdf_path), topics=topics)
            if pdf_path.is_file()
            else self.embed_paper_from_dir(str(pdf_path), topics=topics)
        )

        self.vector_db.add(texts=chunks, embeddings=embeddings, metadatas=metadatas)

        print(f"✅ 已索引论文: {pdf_path}")

    def embed_paper_from_file(self, pdf_path: str, topics: str = None):
        pdf_path = Path(pdf_path)
        topics = topics.split(",") if topics else []
        text = load_pdf_text(pdf_path)
        chunks = [text[i:i+1024] for i in range(0, len(text), 500)]
        embeddings = self.make_request(chunks, topics=topics, title=pdf_path.stem)
        if len(topics) == 0:
            topics = self.get_topic(embeddings, search_top_k=3, num_return=1)
        save_paths = self.save_paper(str(pdf_path), topics=topics)
        metadatas = [{"path": save_paths, "topics": topics}] * len(chunks)
        contents = [f"Title: {pdf_path.stem}\nContent: {chunk}" for chunk in chunks]
        return contents, embeddings, metadatas
    
    def get_topic(self, embeddings, search_top_k: int = 3, num_return: int = 1) -> List[str]:
        topic_counts = {}
        for emb in embeddings:
            search_results = self.search(embedding=emb, top_k=search_top_k)
            for search_result in search_results:
                for topic in search_result["topics"]:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        top_topics = [topic for topic, count in sorted_topics[:num_return]]
        if len(top_topics) == 0:
            top_topics = ["unknown"]
        return top_topics
    
    def embed_paper_from_dir(self, pdf_dir: str, topics=None):
        pdf_paths = load_papers_from_dir(pdf_dir)

        all_chunks, all_embeddings, all_metadatas = [], [], []

        for path in pdf_paths:
            chunks, embeddings, metadatas = self.embed_paper_from_file(path, topics=topics)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            all_metadatas.extend(metadatas)

        return all_chunks, all_embeddings, all_metadatas

    def search(self, query=None, embedding=None, top_k=5):
        assert query or embedding, "必须提供查询文本或嵌入向量。"
        q_emb = embedding if embedding else self.make_request([query])[0]
        results = self.vector_db.search(q_emb, top_k=top_k)

        if len(results["documents"][0]) == 0:
            return []

        res = []

        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            res.append({
                "path": meta["path"][0],
                "content": doc,
                "topics": meta["topics"]
            })
        return res
    
    def search_paper(self, query, top_k=5):
        results = self.search(query=query, top_k=top_k)

        seen_paths = set()
        unique_results = []
        for result in results:
            if result["path"] not in seen_paths:
                unique_results.append(result)
                seen_paths.add(result["path"])
        results = unique_results

        if len(results) == 0:
            print("🔍 未找到相关论文。")
            return

        for i, result in enumerate(results):
            print(f"--- 结果 {i + 1} ---")
            print(f"路径: {result['path']}")
            print(f"主题: {', '.join(result['topics'])}")
            print(f"论文内容: {result['content']}...")
            print()

    def save_paper(self, pdf_path: str, topics: List[str]) -> List[str]:
        pdf_path = Path(pdf_path)
        save_file_name = hashlib.md5(str(pdf_path).encode()).hexdigest() + pdf_path.suffix
        save_paths = []
        for topic in topics:
            save_dir = self.save_dir / topic
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / save_file_name
            copyfile(pdf_path, save_path)
            save_paths.append(str(save_path))
        return save_paths
    
    def make_request(self, texts: List[str], topics: List[str] = None, title: str = None):
        topics = f"Topics: {','.join(topics)}" if topics else ""
        title = f"Title: {title}" if title else ""
        content_prefix = ""
        if topics:
            content_prefix += topics + "\n"
        if title:
            content_prefix += title + "\n"
        if content_prefix:
            texts = [f"Passage: {text}" for text in texts]
            input_texts = [content_prefix + text for text in texts]
        else:
            input_texts = texts

        emb = [
            self.embedder.embed_text(input_text)
            for input_text in tqdm(input_texts, desc="Generating embeddings for papers")
        ]
        return emb