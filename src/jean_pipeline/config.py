from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int
    data_dir: Path
    output_dir: Path
    expected_gpu: str
    train_frac: float
    val_frac: float
    test_frac: float
    box_size_px: int
    box_size_candidates: list[int]
    min_box_size_px: int
    epochs: int
    model_compare_epochs: int
    ablation_epochs: int
    patience: int
    model_compare_patience: int
    ablation_patience: int
    workers: int
    max_det: int
    predict_conf: float
    predict_iou: float
    operating_conf_grid: list[float]
    iou_thresholds: list[float]
    bootstrap_replicates: int
    train_attempts: list[dict]
    ablation_train_attempts: list[dict]
    model_compare_attempts: dict[str, list[dict]]

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def annotations_csv(self) -> Path:
        return self.data_dir / "classification" / "classifications.csv"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def yolo_dir(self) -> Path:
        return self.output_dir / "yolo_dataset"

    @property
    def runs_dir(self) -> Path:
        return self.output_dir / "runs"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def metadata_csv(self) -> Path:
        return self.output_dir / "metadata_cells_detection.csv"

    @property
    def split_summary_csv(self) -> Path:
        return self.output_dir / "split_summary.csv"

    @property
    def data_yaml(self) -> Path:
        return self.yolo_dir / "data.yaml"

    @property
    def best_weights(self) -> Path:
        return self.checkpoints_dir / "best_cell_detector.pt"


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[1]
    raw["data_dir"] = (root / raw["data_dir"]).resolve()
    raw["output_dir"] = (root / raw["output_dir"]).resolve()
    return Config(**raw)


def ensure_dirs(config: Config) -> None:
    for path in [
        config.output_dir,
        config.metrics_dir,
        config.checkpoints_dir,
        config.yolo_dir,
        config.runs_dir,
        config.figures_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
