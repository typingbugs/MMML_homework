import argparse
from dotenv import load_dotenv
from utils.config_loader import load_config
from infrastructure.embeddings import (
    OpenAITextEmbedding, 
    ArkTextEmbedding, 
    ArkImageEmbedding
)
from infrastructure.vector_db.chroma import ChromaDB
from app.paper_service import PaperService
from app.image_service import ImageService


def paper_main(cfg, args):
    embedder = OpenAITextEmbedding(
        base_url=cfg["model"]["paper"]["text"]["base_url"],
        model=cfg["model"]["paper"]["text"]["name"]
    )

    vector_db = ChromaDB(cfg["vector_db"]["persist_dir"], collection_name="papers")
    service = PaperService(embedder, vector_db, save_dir=cfg["file_dir"]["paper"])

    if args.cmd == "add_paper":
        service.add_paper(args.path, args.topics)
    elif args.cmd == "search_paper":
        service.search_paper(args.query, top_k=args.top_k)


def image_main(cfg, args):
    image_embedder = ArkImageEmbedding(
        base_url=cfg["model"]["image"]["image"]["base_url"],
        model=cfg["model"]["image"]["image"]["name"]
    )

    text_setting = cfg["model"]["image"]["text"]
    TextEmbeddingCls = (
        OpenAITextEmbedding 
        if text_setting["service"] == "openai" 
        else ArkTextEmbedding
    )
    text_embedder = TextEmbeddingCls(
        base_url=text_setting["base_url"],
        model=text_setting["name"]
    )

    vector_db = ChromaDB(cfg["vector_db"]["persist_dir"], collection_name="images")
    service = ImageService(image_embedder, text_embedder, vector_db, save_dir=cfg["file_dir"]["image"])

    if args.cmd == "add_image":
        service.add_image(args.path, args.topics)
    elif args.cmd == "search_image":
        service.search_image(args.query, top_k=args.top_k)


def get_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    paper_add_cmd = sub.add_parser("add_paper")
    paper_add_cmd.add_argument("path")
    paper_add_cmd.add_argument("--topics", type=str, help="Comma-separated list of topics to filter papers")

    paper_search_cmd = sub.add_parser("search_paper")
    paper_search_cmd.add_argument("query")
    paper_search_cmd.add_argument("--top_k", type=int, default=5, help="Number of top results to return")

    image_add_cmd = sub.add_parser("add_image")
    image_add_cmd.add_argument("path")
    image_add_cmd.add_argument("--topics", type=str, help="Comma-separated list of topics to filter images")

    image_search_cmd = sub.add_parser("search_image")
    image_search_cmd.add_argument("query")
    image_search_cmd.add_argument("--top_k", type=int, default=5, help="Number of top results to return")

    args = parser.parse_args()
    return args


def main():
    cfg = load_config()
    args = get_args()
    load_dotenv()

    if "paper" in args.cmd:
        paper_main(cfg, args)
        return
    elif "image" in args.cmd:
        image_main(cfg, args)
        return
    

if __name__ == "__main__":
    main()
