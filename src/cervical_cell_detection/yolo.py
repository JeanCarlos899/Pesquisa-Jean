from __future__ import annotations

import gc
import json
import shutil
import subprocess
from pathlib import Path

from .config import Config, ensure_dirs


def check_environment(config: Config) -> None:
    try:
        import torch
        from ultralytics import YOLO  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Create .venv and install requirements-local.txt before running the pipeline."
        ) from exc

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"gpu={name}")
        if config.expected_gpu.lower() not in name.lower():
            raise RuntimeError(f"Unexpected GPU: {name}. Expected to contain: {config.expected_gpu}")
    else:
        raise RuntimeError("CUDA is not available. This pipeline is configured for GPU execution.")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
    except FileNotFoundError:
        print("nvidia-smi not found; continuing with torch CUDA check only.")


def find_existing_best(config: Config, attempt: dict) -> tuple[Path, Path] | None:
    matches = []
    expected = config.runs_dir / str(attempt["name"]) / "weights" / "best.pt"
    if expected.exists():
        matches.append(expected)
    matches.extend(
        config.runs_dir.glob(f"{attempt['name']}*/weights/best.pt"),
    )
    unique = sorted(set(matches), key=lambda path: path.stat().st_mtime, reverse=True)
    if unique:
        return unique[0].parent.parent, unique[0]
    return None


def completed_epochs(save_dir: Path) -> int:
    results = save_dir / "results.csv"
    if not results.exists():
        return 0
    try:
        lines = [line for line in results.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except OSError:
        return 0
    if len(lines) <= 1:
        return 0
    last = lines[-1].split(",", 1)[0].strip()
    try:
        return int(float(last))
    except ValueError:
        return len(lines) - 1


def existing_checkpoint_info(config: Config) -> tuple[Path | None, int]:
    metadata = config.metrics_dir / "treino_detector_config.json"
    if not metadata.exists():
        return None, 0
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, 0
    source = Path(payload.get("source_best_weights", ""))
    source_epochs = int(payload.get("source_epochs", 0) or 0)
    if source_epochs <= 0:
        save_dir = Path(payload.get("save_dir", ""))
        if save_dir.exists():
            source_epochs = completed_epochs(save_dir)
    if source.exists():
        return source, source_epochs
    return None, source_epochs


def save_training_artifact(config: Config, attempt: dict, save_dir: Path, best: Path, source_epochs: int | None = None) -> Path:
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found: {best}")
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, config.best_weights)
    epochs_done = completed_epochs(save_dir) if source_epochs is None else source_epochs
    payload = {
        "best_weights": str(config.best_weights),
        "source_best_weights": str(best),
        "save_dir": str(save_dir),
        "attempt": attempt,
        "box_size_px": config.box_size_px,
        "epochs": config.epochs,
        "source_epochs": epochs_done,
        "patience": config.patience,
    }
    (config.metrics_dir / "treino_detector_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Best detector copied to: {config.best_weights}")
    return config.best_weights


def train_detector(config: Config, force: bool = False) -> Path:
    ensure_dirs(config)
    if config.best_weights.exists() and not force:
        source, source_epochs = existing_checkpoint_info(config)
        if source is None or source_epochs >= config.epochs:
            print(f"Using existing detector weights: {config.best_weights}")
            return config.best_weights
        print(f"Extending existing detector from {source_epochs} to {config.epochs} epochs: {source}")
        return extend_detector(config, source, source_epochs)
    if not config.data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found. Run prepare first: {config.data_yaml}")

    import torch
    from ultralytics import YOLO

    last_error: Exception | None = None
    for idx, attempt in enumerate(config.train_attempts, start=1):
        print("=" * 88)
        print(f"Training attempt {idx}/{len(config.train_attempts)}: {attempt}")
        print("=" * 88)
        if not force:
            existing = find_existing_best(config, attempt)
            if existing is not None:
                save_dir, best = existing
                epochs_done = completed_epochs(save_dir)
                if epochs_done < config.epochs:
                    print(f"Recovering and extending YOLO run from {epochs_done} to {config.epochs} epochs: {best}")
                    return extend_detector(config, best, epochs_done)
                print(f"Recovering completed YOLO run: {best}")
                return save_training_artifact(config, attempt, save_dir, best, source_epochs=epochs_done)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = YOLO(attempt["model"])
            model.train(
                data=str(config.data_yaml),
                project=str(config.runs_dir),
                name=attempt["name"],
                epochs=config.epochs,
                patience=config.patience,
                imgsz=int(attempt["imgsz"]),
                batch=int(attempt["batch"]),
                device=0,
                workers=config.workers,
                pretrained=True,
                optimizer="AdamW",
                lr0=0.002,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5,
                cos_lr=True,
                amp=True,
                cache=bool(attempt.get("cache", False)),
                seed=config.seed,
                deterministic=False,
                single_cls=True,
                plots=bool(attempt.get("plots", True)),
                save=True,
                save_period=25,
                exist_ok=True,
                close_mosaic=25,
                hsv_h=0.015,
                hsv_s=0.45,
                hsv_v=0.25,
                degrees=180.0,
                translate=0.10,
                scale=0.35,
                shear=3.0,
                perspective=0.0005,
                fliplr=0.50,
                flipud=0.50,
                mosaic=float(attempt.get("mosaic", 0.35)),
                mixup=float(attempt.get("mixup", 0.0)),
                erasing=0.15,
                box=8.0,
                cls=0.25,
                dfl=1.5,
                nbs=64,
            )
            trainer = getattr(model, "trainer", None)
            save_dir = Path(getattr(trainer, "save_dir", config.runs_dir / str(attempt["name"])))
            best = save_dir / "weights" / "best.pt"
            return save_training_artifact(config, attempt, save_dir, best)
        except RuntimeError as exc:
            last_error = exc
            message = str(exc).lower()
            print(f"Attempt failed: {exc}")
            if "out of memory" in message or "cuda" in message:
                try:
                    del model
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"All training attempts failed. Last error: {last_error}")


def extend_detector(config: Config, weights: Path, completed: int) -> Path:
    import torch
    from ultralytics import YOLO

    remaining = max(config.epochs - completed, 1)
    attempt = dict(config.train_attempts[0])
    attempt["name"] = f"{attempt['name']}_extend_to_{config.epochs:03d}"
    attempt["model"] = str(weights)
    print(f"Additional epochs: {remaining} | attempt: {attempt}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = YOLO(str(weights))
    model.train(
        data=str(config.data_yaml),
        project=str(config.runs_dir),
        name=attempt["name"],
        epochs=remaining,
        patience=max(4, min(config.patience, remaining)),
        imgsz=int(attempt["imgsz"]),
        batch=int(attempt["batch"]),
        device=0,
        workers=config.workers,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,
        cos_lr=True,
        amp=True,
        cache=bool(attempt.get("cache", False)),
        seed=config.seed,
        deterministic=False,
        single_cls=True,
        plots=bool(attempt.get("plots", True)),
        save=True,
        save_period=25,
        exist_ok=True,
        close_mosaic=5,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.25,
        degrees=180.0,
        translate=0.10,
        scale=0.30,
        shear=3.0,
        perspective=0.0005,
        fliplr=0.50,
        flipud=0.50,
        mosaic=float(attempt.get("mosaic", 0.20)),
        mixup=float(attempt.get("mixup", 0.0)),
        erasing=0.10,
        box=8.0,
        cls=0.25,
        dfl=1.5,
        nbs=64,
    )
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", config.runs_dir / str(attempt["name"])))
    best = save_dir / "weights" / "best.pt"
    return save_training_artifact(config, attempt, save_dir, best, source_epochs=config.epochs)


def predict_split(config: Config, split: str, force: bool = False) -> Path:
    ensure_dirs(config)
    out_csv = config.output_dir / f"predicoes_{split}.csv"
    if out_csv.exists() and not force:
        print(f"Using existing predictions: {out_csv}")
        return out_csv
    if not config.best_weights.exists():
        raise FileNotFoundError(f"Detector weights not found. Run train first: {config.best_weights}")

    import pandas as pd
    from ultralytics import YOLO

    source_dir = config.yolo_dir / "images" / split
    if not source_dir.exists():
        raise FileNotFoundError(f"Split image directory not found: {source_dir}")
    image_names = {path.stem: path.name for path in source_dir.iterdir() if path.is_file()}
    if not image_names:
        raise FileNotFoundError(f"No images found for split: {source_dir}")
    exported: list[dict] = []

    def capture_predictions(validator) -> None:
        exported.extend(validator.jdict)

    model = YOLO(str(config.best_weights))
    model.add_callback("on_val_end", capture_predictions)
    model.val(
        data=str(config.data_yaml),
        split=split,
        imgsz=int(config.train_attempts[0]["imgsz"]),
        batch=int(config.train_attempts[0]["batch"]),
        conf=config.predict_conf,
        iou=config.predict_iou,
        max_det=config.max_det,
        device=0,
        workers=config.workers,
        augment=False,
        plots=False,
        save_json=True,
        project=str(config.runs_dir),
        name=f"metrics_export_{split}",
        exist_ok=True,
        verbose=False,
    )
    rows = []
    detection_ids: dict[str, int] = {}
    for item in exported:
        stem = str(item["image_id"])
        image_name = image_names.get(stem)
        if image_name is None:
            continue
        x1, y1, width, height = [float(value) for value in item["bbox"]]
        detection_ids[image_name] = detection_ids.get(image_name, 0) + 1
        rows.append(
            {
                "particao": split,
                "arquivo_imagem": image_name,
                "caminho_imagem_yolo": str(source_dir / image_name),
                "det_id": detection_ids[image_name],
                "conf": float(item["score"]),
                "pred_x1": x1,
                "pred_y1": y1,
                "pred_x2": x1 + width,
                "pred_y2": y1 + height,
                "pred_w": width,
                "pred_h": height,
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Predictions saved to: {out_csv} ({len(rows)} detections)")
    return out_csv
