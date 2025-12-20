from pathlib import Path

def load_images_from_dir(dir_path):
    dir_path = Path(dir_path)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return [
        str(dir_path / f)
        for f in dir_path.iterdir()
        if f.suffix.lower() in exts
    ]
