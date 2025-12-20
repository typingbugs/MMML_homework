import argparse
import os
from utils.config_loader import load_config
from infrastructure.embeddings.openai_text import OpenAITextEmbedding
from infrastructure.embeddings.openai_multimodal import OpenAIMultimodalEmbedding
from infrastructure.vector_db.chroma import ChromaDB
from app.paper_service import PaperService
from app.image_service import ImageService


def paper_main(cfg, args, api_key):
    embedder = OpenAITextEmbedding(
        api_key=api_key,
        base_url=cfg["openai"]["base_url"],
        model=cfg["embedding_model"]["text"]
    )

    vector_db = ChromaDB(cfg["vector_db"]["persist_dir"], collection_name="papers")
    service = PaperService(embedder, vector_db, save_dir=cfg["file_dir"]["paper"])

    if args.cmd == "add_paper":
        service.add_paper(args.path, args.topics)
    elif args.cmd == "search_paper":
        service.search(args.query, top_k=args.top_k)


def image_main(cfg, args, api_key):
    embedder = OpenAIMultimodalEmbedding(
        api_key=api_key,
        base_url=cfg["openai"]["base_url"],
        model=cfg["embedding_model"]["multimodal"]
    )   

    vector_db = ChromaDB(cfg["vector_db"]["persist_dir"], collection_name="images")
    service = ImageService(embedder, vector_db, save_dir=cfg["file_dir"]["image"])

    if args.cmd == "add_images":
        service.add_image(args.path, args.topics)
    elif args.cmd == "search_images":
        service.search(args.query, top_k=args.top_k)




def get_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    paper_add_cmd = sub.add_parser("add_paper")
    paper_add_cmd.add_argument("path")
    paper_add_cmd.add_argument("--topics", type=str, help="Comma-separated list of topics to filter papers")

    paper_search_cmd = sub.add_parser("search_paper")
    paper_search_cmd.add_argument("query")
    paper_search_cmd.add_argument("--top_k", type=int, default=5, help="Number of top results to return")

    image_add_cmd = sub.add_parser("add_images")
    image_add_cmd.add_argument("path")
    image_add_cmd.add_argument("--topics", type=str, help="Comma-separated list of topics to filter images")

    image_search_cmd = sub.add_parser("search_images")
    image_search_cmd.add_argument("query")
    image_search_cmd.add_argument("--top_k", type=int, default=5, help="Number of top results to return")

    args = parser.parse_args()
    return args


def main():
    cfg = load_config()
    args = get_args()

    api_key = os.getenv("OPENAI_API_KEY")

    if "paper" in args.cmd:
        paper_main(cfg, args, api_key)
        return
    elif "image" in args.cmd:
        image_main(cfg, args, api_key)
        return
    

if __name__ == "__main__":
    main()
