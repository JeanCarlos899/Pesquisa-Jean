from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlencode


FIGSHARE_API = "https://api.figshare.com/v2"
CRIC_COLLECTION_ID = 4960286
CLASSIFICATION_ARTICLE_ID = 12233156


def fetch_json(url: str) -> dict | list:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, target: Path, expected_md5: str | None = None, force: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        if expected_md5 and file_md5(target) != expected_md5:
            print(f"[warn] checksum mismatch, redownloading: {target}")
        else:
            print(f"[skip] {target}")
            return

    temporary = target.with_suffix(target.suffix + ".download")
    urllib.request.urlretrieve(url, temporary)
    if expected_md5:
        actual = file_md5(temporary)
        if actual != expected_md5:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"MD5 mismatch for {target.name}: expected {expected_md5}, got {actual}")
    temporary.replace(target)
    print(f"[ok]   {target}")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_collection_articles(collection_id: int) -> list[dict]:
    articles: list[dict] = []
    page = 1
    page_size = 100
    while True:
        query = urlencode({"page": page, "page_size": page_size})
        payload = fetch_json(f"{FIGSHARE_API}/collections/{collection_id}/articles?{query}")
        batch = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not batch:
            break
        articles.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return articles


def image_number(title: str) -> int | None:
    match = re.search(r"#(\d+)\s*$", title)
    return int(match.group(1)) if match else None


def download_cric_cervix(data_dir: Path, force: bool = False) -> None:
    data_dir = data_dir.resolve()
    classification_dir = data_dir / "classification"
    images_dir = data_dir / "images"

    print("Downloading CRIC Cervix classification files from Figshare...")
    classification = fetch_json(f"{FIGSHARE_API}/articles/{CLASSIFICATION_ARTICLE_ID}")
    for item in classification["files"]:
        checksum = item.get("computed_md5") or item.get("supplied_md5") or None
        download_file(item["download_url"], classification_dir / item["name"], checksum, force=force)

    print("Downloading CRIC Cervix images from Figshare collection...")
    articles = list_collection_articles(CRIC_COLLECTION_ID)
    image_articles = [
        article
        for article in articles
        if article.get("defined_type_name") == "figure" and image_number(article.get("title", "")) is not None
    ]
    image_articles.sort(key=lambda article: image_number(article["title"]) or 0)
    if len(image_articles) != 400:
        raise RuntimeError(f"Expected 400 CRIC image articles, found {len(image_articles)}")

    for article in image_articles:
        number = image_number(article["title"])
        details = fetch_json(article["url_public_api"])
        files = details.get("files", [])
        if len(files) != 1:
            raise RuntimeError(f"Expected one file for {article['title']}, found {len(files)}")
        item = files[0]
        checksum = item.get("computed_md5") or item.get("supplied_md5") or None
        filename = f"cric_image_{number:03d}_{item['name']}"
        download_file(item["download_url"], images_dir / filename, checksum, force=force)

    csv_path = classification_dir / "classifications.csv"
    image_count = len(list(images_dir.glob("*.png")))
    if not csv_path.exists() or image_count != 400:
        raise RuntimeError(f"Download incomplete: csv={csv_path.exists()} images={image_count}")
    print(f"CRIC Cervix ready at {data_dir} ({image_count} images).")
