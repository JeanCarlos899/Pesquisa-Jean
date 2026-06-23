from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from .data import prepare_dataset
from .materials import make_materials
from .metrics import evaluate_detector, finalize_test_metrics
from .yolo import check_environment, predict_split, train_detector


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/local_3060.json")
    parser.add_argument("--force", action="store_true")


def cmd_check(args) -> None:
    config = load_config(args.config)
    check_environment(config)
    print("JEAN_ENV_READY")


def cmd_prepare(args) -> None:
    config = load_config(args.config)
    prepare_dataset(config, force=args.force)


def cmd_train(args) -> None:
    config = load_config(args.config)
    prepare_dataset(config, force=False)
    train_detector(config, force=args.force)


def cmd_predict(args) -> None:
    config = load_config(args.config)
    predict_split(config, args.split, force=args.force)


def cmd_eval(args) -> None:
    config = load_config(args.config)
    predict_split(config, args.split, force=args.force)
    evaluate_detector(config, args.split, force=args.force)
    if args.split == "test":
        if not (config.metrics_dir / "avaliacao_val_limiares.csv").exists():
            predict_split(config, "val", force=False)
            evaluate_detector(config, "val", force=False)
        finalize_test_metrics(config, force=args.force)


def cmd_finalize(args) -> None:
    config = load_config(args.config)
    finalize_test_metrics(config, force=args.force)


def cmd_materials(args) -> None:
    base = load_config(args.config)
    config = base
    if args.model != "main":
        config = model_compare_config(base, args.model)
        config.metrics_dir.mkdir(parents=True, exist_ok=True)
        for name in ["bbox_sweep_summary.csv", "model_compare_summary.csv"]:
            source = base.metrics_dir / name
            target = config.metrics_dir / name
            if source.exists():
                shutil.copy2(source, target)
    make_materials(config, output_dir=args.output)


def cmd_all(args) -> None:
    config = load_config(args.config)
    prepare_dataset(config, force=args.force)
    train_detector(config, force=args.force)
    for split in ["val", "test"]:
        predict_split(config, split, force=args.force)
        evaluate_detector(config, split, force=args.force)
    finalize_test_metrics(config, force=args.force)
    make_materials(config, output_dir=args.output)


def cmd_bbox_sweep(args) -> None:
    base = load_config(args.config)
    rows = []
    start = time.time()
    for size in base.box_size_candidates:
        print("=" * 88)
        print(f"BBox pseudo-GT ablation: {size}px")
        print("=" * 88)
        sweep_config: Config = replace(
            base,
            box_size_px=int(size),
            output_dir=base.output_dir / "bbox_sweep" / f"box_{int(size):03d}",
            epochs=base.ablation_epochs,
            patience=base.ablation_patience,
            train_attempts=base.ablation_train_attempts,
        )
        prepare_dataset(sweep_config, force=args.force)
        train_detector(sweep_config, force=args.force)
        predict_split(sweep_config, "val", force=args.force)
        evaluate_detector(sweep_config, "val", force=args.force)
        val = pd.read_csv(sweep_config.metrics_dir / "avaliacao_val_limiares.csv")
        best = val[val["iou"].eq(0.50)].sort_values(["f1", "revocacao", "precisao"], ascending=False).iloc[0]
        ap = pd.read_csv(sweep_config.metrics_dir / "map_val.csv")
        ap50 = float(ap.loc[ap["iou"].eq(0.50), "ap"].iloc[0])
        rows.append(
            {
                "box_size_px": int(size),
                "best_conf_val_iou50": float(best["conf"]),
                "f1_val_iou50": float(best["f1"]),
                "precisao_val_iou50": float(best["precisao"]),
                "revocacao_val_iou50": float(best["revocacao"]),
                "mae_contagem_val": float(best["mae_contagem"]),
                "ap50_val": ap50,
                "output_dir": str(sweep_config.output_dir),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["f1_val_iou50", "ap50_val"], ascending=False)
    base.metrics_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(base.metrics_dir / "bbox_sweep_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"BBox sweep time: {(time.time() - start) / 60:.1f} min")


def best_training_metrics(run_dir: Path) -> dict:
    rows = {}
    for results_path in sorted(run_dir.glob("*/results.csv")):
        try:
            table = pd.read_csv(results_path)
        except Exception:
            continue
        table.columns = [column.strip() for column in table.columns]
        if "metrics/mAP50(B)" not in table.columns:
            continue
        best = table.iloc[table["metrics/mAP50(B)"].idxmax()]
        rows[results_path.parent.name] = {
            "train_run": results_path.parent.name,
            "train_best_epoch": int(best["epoch"]),
            "train_best_precision": float(best.get("metrics/precision(B)", 0.0)),
            "train_best_recall": float(best.get("metrics/recall(B)", 0.0)),
            "train_best_map50": float(best.get("metrics/mAP50(B)", 0.0)),
            "train_best_map50_95": float(best.get("metrics/mAP50-95(B)", 0.0)),
        }
    if not rows:
        return {}
    return max(rows.values(), key=lambda item: item["train_best_map50"])


def model_summary_row(config: Config, model_label: str, force: bool = False) -> dict:
    predict_split(config, "val", force=force)
    evaluate_detector(config, "val", force=force)
    val = pd.read_csv(config.metrics_dir / "avaliacao_val_limiares.csv")
    best = val[val["iou"].eq(0.50)].sort_values(["f1", "revocacao", "precisao"], ascending=False).iloc[0]
    ap = pd.read_csv(config.metrics_dir / "map_val.csv")
    row = {
        "modelo": model_label,
        "box_size_px": config.box_size_px,
        "best_conf_val_iou50": float(best["conf"]),
        "f1_val_iou50": float(best["f1"]),
        "precisao_val_iou50": float(best["precisao"]),
        "revocacao_val_iou50": float(best["revocacao"]),
        "mae_contagem_val": float(best["mae_contagem"]),
        "ap50_val_custom": float(ap.loc[ap["iou"].eq(0.50), "ap"].iloc[0]),
        "output_dir": str(config.output_dir),
    }
    row.update(best_training_metrics(config.runs_dir))
    return row


def cmd_model_compare(args) -> None:
    base = load_config(args.config)
    rows = []
    start = time.time()
    requested = [model.lower() for model in args.models]
    if base.best_weights.exists():
        print("=" * 88)
        print("Model capacity comparison baseline: YOLO11n")
        print("=" * 88)
        rows.append(model_summary_row(base, "YOLO11n", force=False))
    for model_name in requested:
        if model_name not in base.model_compare_attempts:
            raise ValueError(f"Modelo desconhecido para comparacao: {model_name}. Use s ou m.")
        print("=" * 88)
        print(f"Model capacity comparison: YOLO11{model_name} | bbox={base.box_size_px}px")
        print("=" * 88)
        compare_config = model_compare_config(base, model_name)
        prepare_dataset(compare_config, force=args.force)
        if args.test and compare_config.best_weights.exists() and not args.force:
            print(f"Using frozen checkpoint for test evaluation: {compare_config.best_weights}")
        else:
            train_detector(compare_config, force=args.force)
        if args.test:
            predict_split(compare_config, "test", force=args.force)
            evaluate_detector(compare_config, "test", force=args.force)
            finalize_test_metrics(compare_config, force=args.force)

        rows.append(model_summary_row(compare_config, f"YOLO11{model_name}", force=args.force))

    out = base.metrics_dir / "model_compare_summary.csv"
    base.metrics_dir.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame(rows)
    if out.exists() and not args.force:
        previous = pd.read_csv(out)
        current = pd.concat([previous[~previous["modelo"].isin(current["modelo"])], current], ignore_index=True)
    current = current.sort_values(["f1_val_iou50", "ap50_val_custom"], ascending=False)
    current.to_csv(out, index=False)
    print(current.to_string(index=False))
    print(f"Model comparison time: {(time.time() - start) / 60:.1f} min")


def model_compare_config(base: Config, model_name: str) -> Config:
    model_name = model_name.lower()
    if model_name not in base.model_compare_attempts:
        raise ValueError(f"Modelo desconhecido para comparacao: {model_name}. Use s ou m.")
    return replace(
        base,
        output_dir=base.output_dir / "model_compare" / f"yolo11{model_name}",
        epochs=base.model_compare_epochs,
        patience=base.model_compare_patience,
        train_attempts=base.model_compare_attempts[model_name],
    )


def cmd_status(args) -> None:
    config = load_config(args.config)
    artifacts = [
        ("metadata", config.metadata_csv),
        ("data_yaml", config.data_yaml),
        ("best_weights", config.best_weights),
        ("pred_val", config.output_dir / "predicoes_val.csv"),
        ("eval_val", config.metrics_dir / "avaliacao_val_limiares.csv"),
        ("pred_test", config.output_dir / "predicoes_test.csv"),
        ("eval_test", config.metrics_dir / "avaliacao_test_limiares.csv"),
        ("operational_test", config.metrics_dir / "metricas_operacionais_teste.csv"),
        ("bbox_sweep", config.metrics_dir / "bbox_sweep_summary.csv"),
        ("model_compare", config.metrics_dir / "model_compare_summary.csv"),
    ]
    for name, path in artifacts:
        if path.exists():
            size_mb = path.stat().st_size / 1024**2
            print(f"[ok]       {name:18s} {path} ({size_mb:.2f} MB)")
        else:
            print(f"[missing]  {name:18s} {path}")


def cmd_package(args) -> None:
    config = load_config(args.config)
    target = shutil.make_archive(
        str(config.output_dir.with_name("outputs_detection_package")),
        "zip",
        root_dir=config.output_dir.parent,
        base_dir=config.output_dir.name,
    )
    print(f"Package saved to: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jean CRIC single-class cell detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "check": cmd_check,
        "status": cmd_status,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "predict": cmd_predict,
        "eval": cmd_eval,
        "finalize": cmd_finalize,
        "bbox-sweep": cmd_bbox_sweep,
        "model-compare": cmd_model_compare,
        "materials": cmd_materials,
        "all": cmd_all,
        "package": cmd_package,
    }
    for name, fn in commands.items():
        command = sub.add_parser(name)
        add_common(command)
        if name in {"predict", "eval"}:
            command.add_argument("--split", choices=["train", "val", "test"], default="test")
        if name in {"materials", "all"}:
            command.add_argument("--output", default="materiais_artigo")
        if name == "materials":
            command.add_argument("--model", choices=["main", "s", "m"], default="main")
        if name == "model-compare":
            command.add_argument("--models", nargs="+", default=["s"], choices=["s", "m"])
            command.add_argument("--test", action="store_true")
        command.set_defaults(func=fn)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
