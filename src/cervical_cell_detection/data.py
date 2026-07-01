from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

from .config import Config, ensure_dirs


REQUIRED_COLUMNS = {
    "image_id",
    "image_filename",
    "cell_id",
    "bethesda_system",
    "nucleus_x",
    "nucleus_y",
}


def image_files(config: Config) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted(path for path in config.images_dir.glob("*") if path.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(f"No images found in {config.images_dir}")
    return files


def load_raw_annotations(config: Config) -> pd.DataFrame:
    if not config.annotations_csv.exists():
        raise FileNotFoundError(f"CRIC annotations not found: {config.annotations_csv}")
    table = pd.read_csv(config.annotations_csv)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns in CRIC CSV: {sorted(missing)}")
    table["nucleus_x"] = pd.to_numeric(table["nucleus_x"], errors="coerce")
    table["nucleus_y"] = pd.to_numeric(table["nucleus_y"], errors="coerce")
    table = table.dropna(subset=["nucleus_x", "nucleus_y"]).copy()
    table["class_id"] = 0
    table["class_name"] = "cell"
    return table


def resolve_image_paths(config: Config, annotations: pd.DataFrame) -> pd.DataFrame:
    files = image_files(config)
    by_name = {path.name: path for path in files}
    resolved: list[Path | None] = []
    for row in annotations.itertuples(index=False):
        filename = str(row.image_filename)
        direct = by_name.get(filename)
        if direct is not None:
            resolved.append(direct)
            continue
        matches = sorted(path for path in files if path.name.endswith(filename))
        resolved.append(matches[0] if matches else None)
    table = annotations.copy()
    table["image_path"] = resolved
    bad = table[table["image_path"].isna()]
    if len(bad):
        sample = bad[["image_id", "image_filename"]].head().to_dict("records")
        raise ValueError(f"{len(bad)} annotations did not match images. Examples: {sample}")
    table["image_path"] = table["image_path"].map(lambda path: str(Path(path)))
    return table


def add_image_sizes(annotations: pd.DataFrame) -> pd.DataFrame:
    sizes: dict[str, tuple[int, int]] = {}
    for path in tqdm(sorted(annotations["image_path"].unique()), desc="Reading image sizes"):
        with Image.open(path) as image:
            sizes[path] = image.size
    table = annotations.copy()
    table["width"] = table["image_path"].map(lambda path: sizes[path][0])
    table["height"] = table["image_path"].map(lambda path: sizes[path][1])
    outside = table[
        (table["nucleus_x"] < 1)
        | (table["nucleus_x"] > table["width"])
        | (table["nucleus_y"] < 1)
        | (table["nucleus_y"] > table["height"])
    ]
    if len(outside):
        sample = outside[["image_id", "cell_id", "nucleus_x", "nucleus_y", "width", "height"]].head().to_dict("records")
        raise ValueError(f"Coordinates outside image limits: {sample}")
    return table


def balanced_image_split(config: Config, annotations: pd.DataFrame) -> dict[str, set[str]]:
    if abs(config.train_frac + config.val_frac + config.test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    counts = annotations.groupby("image_path").size().rename("objects").reset_index()
    counts = counts.sample(frac=1.0, random_state=config.seed).sort_values("objects", ascending=False).reset_index(drop=True)
    n_images = len(counts)
    targets_n = {
        "train": int(round(n_images * config.train_frac)),
        "val": int(round(n_images * config.val_frac)),
    }
    targets_n["test"] = n_images - targets_n["train"] - targets_n["val"]
    total_objects = float(counts["objects"].sum())
    targets_objects = {
        "train": total_objects * config.train_frac,
        "val": total_objects * config.val_frac,
        "test": total_objects * config.test_frac,
    }
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    object_sum = {"train": 0, "val": 0, "test": 0}
    for row in counts.itertuples(index=False):
        candidates = [split for split in buckets if len(buckets[split]) < targets_n[split]]
        chosen = min(candidates, key=lambda split: (object_sum[split] + row.objects) / max(targets_objects[split], 1.0))
        buckets[chosen].append(row.image_path)
        object_sum[chosen] += int(row.objects)
    return {split: set(paths) for split, paths in buckets.items()}


def point_to_box(row, box_size_px: int, min_box_size_px: int) -> tuple[float, float, float, float]:
    size = max(float(box_size_px), float(min_box_size_px))
    half = size / 2.0
    x = float(row.nucleus_x) - 1.0
    y = float(row.nucleus_y) - 1.0
    width = float(row.width)
    height = float(row.height)
    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(width, x + half)
    y2 = min(height, y + half)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box for image={row.image_id}, cell={row.cell_id}")
    return x1, y1, x2, y2


def add_boxes(config: Config, annotations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in annotations.itertuples(index=False):
        x1, y1, x2, y2 = point_to_box(row, config.box_size_px, config.min_box_size_px)
        rows.append((x1, y1, x2, y2))
    table = annotations.copy()
    table[["box_x1", "box_y1", "box_x2", "box_y2"]] = pd.DataFrame(rows, index=table.index)
    table["box_w"] = table["box_x2"] - table["box_x1"]
    table["box_h"] = table["box_y2"] - table["box_y1"]
    return table


def to_yolo_line(row) -> str:
    xc = ((row.box_x1 + row.box_x2) / 2.0) / row.width
    yc = ((row.box_y1 + row.box_y2) / 2.0) / row.height
    bw = (row.box_x2 - row.box_x1) / row.width
    bh = (row.box_y2 - row.box_y1) / row.height
    vals = [xc, yc, bw, bh]
    if not all(0.0 <= value <= 1.0 for value in vals):
        raise ValueError(f"YOLO values outside [0,1]: {vals}")
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def prepare_dataset(config: Config, force: bool = False) -> pd.DataFrame:
    ensure_dirs(config)
    if config.metadata_csv.exists() and config.data_yaml.exists() and not force:
        print(f"Using existing prepared dataset: {config.metadata_csv}")
        return pd.read_csv(config.metadata_csv)

    if config.yolo_dir.exists():
        shutil.rmtree(config.yolo_dir)
    for split in ["train", "val", "test"]:
        (config.yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations = load_raw_annotations(config)
    annotations = resolve_image_paths(config, annotations)
    annotations = add_image_sizes(annotations)
    split_sets = balanced_image_split(config, annotations)
    split_lookup = {path: split for split, paths in split_sets.items() for path in paths}
    annotations["split"] = annotations["image_path"].map(split_lookup)
    if annotations["split"].isna().any():
        raise RuntimeError("Some annotations did not receive a split")
    annotations = add_boxes(config, annotations)
    annotations = annotations.rename(
        columns={
            "image_id": "id_imagem",
            "image_filename": "arquivo_imagem",
            "image_doi": "doi_imagem",
            "cell_id": "id_celula",
            "bethesda_system": "classe_bethesda",
            "nucleus_x": "nucleo_x",
            "nucleus_y": "nucleo_y",
            "image_path": "caminho_imagem",
            "split": "particao",
        }
    )
    annotations["classe_detector"] = "cell"
    annotations["classe_id"] = 0
    annotations["arquivo_yolo"] = annotations["caminho_imagem"].map(lambda path: Path(path).name)

    label_lines: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in annotations.itertuples(index=False):
        label_lines[(row.caminho_imagem, row.particao)].append(to_yolo_line(row))

    for image_path in tqdm(sorted(annotations["caminho_imagem"].unique()), desc="Writing YOLO dataset"):
        split = split_lookup[image_path]
        source = Path(image_path)
        shutil.copy2(source, config.yolo_dir / "images" / split / source.name)
        label_path = config.yolo_dir / "labels" / split / f"{source.stem}.txt"
        lines = label_lines.get((image_path, split), [])
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    try:
        yolo_path = config.yolo_dir.relative_to(Path.cwd()).as_posix()
    except ValueError:
        yolo_path = config.yolo_dir.as_posix()
    data_yaml = "\n".join(
        [
            f"path: {yolo_path}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 1",
            "names:",
            "  0: cell",
            "",
        ]
    )
    config.data_yaml.write_text(data_yaml, encoding="utf-8")
    annotations.to_csv(config.metadata_csv, index=False)
    write_dataset_summaries(config, annotations)
    draw_annotation_audit(config, annotations)
    print(pd.read_csv(config.split_summary_csv).to_string(index=False))
    return annotations


def write_dataset_summaries(config: Config, metadata: pd.DataFrame) -> None:
    summary = (
        metadata.groupby("particao")
        .agg(imagens=("caminho_imagem", "nunique"), celulas=("id_celula", "count"))
        .reindex(["train", "val", "test"])
        .reset_index()
    )
    summary["celulas_por_imagem_media"] = (
        metadata.groupby("particao").size() / metadata.groupby("particao")["caminho_imagem"].nunique()
    ).reindex(["train", "val", "test"]).round(2).to_numpy()
    summary.to_csv(config.split_summary_csv, index=False)

    composition = (
        metadata["classe_bethesda"]
        .value_counts()
        .rename_axis("classe_bethesda")
        .reset_index(name="celulas")
    )
    composition["percentual"] = (100 * composition["celulas"] / composition["celulas"].sum()).round(2)
    composition.to_csv(config.metrics_dir / "composicao_bethesda_original.csv", index=False)

    per_image = metadata.groupby("caminho_imagem").size().rename("celulas").reset_index()
    per_image["particao"] = per_image["caminho_imagem"].map(metadata.drop_duplicates("caminho_imagem").set_index("caminho_imagem")["particao"])
    per_image.to_csv(config.metrics_dir / "contagem_celulas_por_imagem_gt.csv", index=False)


def draw_annotation_audit(config: Config, metadata: pd.DataFrame, samples_per_split: int = 2) -> None:
    random.seed(config.seed)
    selected = []
    for split in ["train", "val", "test"]:
        candidates = sorted(metadata.loc[metadata["particao"] == split, "caminho_imagem"].unique())
        selected.extend((split, path) for path in random.sample(candidates, min(samples_per_split, len(candidates))))
    if not selected:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    for ax, (split, path) in zip(axes, selected):
        subset = metadata[metadata["caminho_imagem"] == path]
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for row in subset.itertuples(index=False):
            draw.rectangle([row.box_x1, row.box_y1, row.box_x2, row.box_y2], outline=(255, 40, 40), width=2)
            draw.ellipse([row.nucleo_x - 3, row.nucleo_y - 3, row.nucleo_x + 3, row.nucleo_y + 3], fill=(40, 255, 80))
        ax.imshow(image)
        ax.set_title(f"{split}: {len(subset)} cells")
        ax.axis("off")
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.figures_dir / "auditoria_boxes_pseudo_gt.png", dpi=180)
    plt.close(fig)


def load_metadata(config: Config) -> pd.DataFrame:
    if not config.metadata_csv.exists():
        return prepare_dataset(config)
    return pd.read_csv(config.metadata_csv)
