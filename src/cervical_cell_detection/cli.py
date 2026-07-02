from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from .data import load_metadata, prepare_kfold_dataset
from .download_data import download_cric_cervix
from .metrics import (
    detection_metrics_from_counts,
    evaluate_detector,
    load_predictions,
    per_image_counts,
    select_operating_threshold,
)
from .yolo import check_environment, predict_split, train_detector


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/local_3060.json")
    parser.add_argument("--force", action="store_true")


def cmd_check(args) -> None:
    config = load_config(args.config)
    check_environment(config)
    print("JEAN_ENV_READY")


def cmd_download_data(args) -> None:
    config = load_config(args.config)
    download_cric_cervix(config.data_dir, force=args.force)


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

def kfold_model_config(base: Config, model_name: str, fold: int, folds: int, epochs: int, patience: int) -> Config:
    model_name = model_name.lower()
    attempts = base.kfold_train_attempts
    if not attempts:
        raise ValueError("kfold_train_attempts must be configured for the current protocol.")
    return replace(
        base,
        output_dir=base.output_dir / "kfold" / f"yolo11{model_name}_box{base.box_size_px}" / f"fold_{fold:02d}",
        epochs=epochs,
        patience=patience,
        train_attempts=attempts,
        kfold_splits=folds,
        kfold_model=model_name,
    )


def fold_operating_row(config: Config, fold: int, model_name: str, selected_threshold: dict, split: str = "test", iou: float = 0.50) -> dict:
    grid = pd.read_csv(config.metrics_dir / f"avaliacao_{split}_limiares.csv")
    grid_iou = grid[grid["iou"].eq(iou)].copy()
    if grid_iou.empty:
        raise ValueError(f"No {split} rows for IoU={iou} in {config.metrics_dir}")
    selected_conf = float(selected_threshold["conf"])
    selected = grid_iou.iloc[(grid_iou["conf"] - selected_conf).abs().argsort()].iloc[0]

    metadata = load_metadata(config)
    gt = metadata[metadata["particao"] == split].copy()
    pred = load_predictions(config.output_dir / f"predicoes_{split}.csv")
    per_image = per_image_counts(gt, pred, float(selected["conf"]), iou)
    per_image["fold"] = fold
    per_image.to_csv(config.metrics_dir / f"contagem_por_imagem_{split}_fold.csv", index=False)
    totals = detection_metrics_from_counts(int(per_image["tp"].sum()), int(per_image["fp"].sum()), int(per_image["fn"].sum()))
    error = per_image["erro_contagem"]

    ap = pd.read_csv(config.metrics_dir / f"map_{split}.csv")
    ap_iou = float(ap.loc[ap["iou"].eq(iou), "ap"].iloc[0])
    train_info = best_training_metrics(config.runs_dir)
    return {
        "fold": fold,
        "modelo": f"YOLO11{model_name}",
        "box_size_px": config.box_size_px,
        "split_avaliado": split,
        "heldout_imagens": int(gt["caminho_imagem"].nunique()),
        "heldout_celulas": int(len(gt)),
        "conf_selecionado_val": float(selected_conf),
        "conf_aplicado_teste": float(selected["conf"]),
        "iou": iou,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precisao": totals["precisao"],
        "revocacao": totals["revocacao"],
        "f1": totals["f1"],
        "mae_contagem": float(error.abs().mean()),
        "rmse_contagem": float((error.pow(2).mean()) ** 0.5),
        "bias_contagem": float(error.mean()),
        "ap50": ap_iou,
        "f1_val_interno": float(selected_threshold["metricas_validacao"]["f1"]),
        "mae_val_interno": float(selected_threshold["metricas_validacao"]["mae_contagem"]),
        "output_dir": str(config.output_dir),
        **train_info,
    }


def simple_markdown(table: pd.DataFrame) -> str:
    data = table.copy()
    lines = [
        "| " + " | ".join(map(str, data.columns)) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]
    for row in data.to_numpy():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines) + "\n"


def write_kfold_summary(rows: list[dict], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(rows).sort_values("fold")
    folds.to_csv(out / "kfold_folds.csv", index=False)
    metric_columns = ["precisao", "revocacao", "f1", "mae_contagem", "rmse_contagem", "bias_contagem", "ap50"]
    summary_rows = []
    for metric in metric_columns:
        values = folds[metric].astype(float)
        summary_rows.append(
            {
                "metrica": metric,
                "media": float(values.mean()),
                "desvio_padrao": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out / "kfold_resumo.csv", index=False)
    (out / "kfold_folds.md").write_text(
        simple_markdown(
            folds[
                [
                    "fold",
                    "split_avaliado",
                    "heldout_imagens",
                    "heldout_celulas",
                    "conf_selecionado_val",
                    "conf_aplicado_teste",
                    "f1",
                    "precisao",
                    "revocacao",
                    "mae_contagem",
                    "ap50",
                    "f1_val_interno",
                ]
            ]
        ),
        encoding="utf-8",
    )
    (out / "kfold_resumo.md").write_text(simple_markdown(pd.DataFrame(summary_rows)), encoding="utf-8")
    readme = [
        "# K-fold box 144",
        "",
        "Validacao cruzada por imagem. Cada fold usa um quinto das imagens como teste externo e separa uma validacao interna apenas dentro dos quatro quintos restantes.",
        "",
        "- `kfold_folds.csv`: metricas por fold no teste externo, usando o limiar escolhido na validacao interna daquele fold.",
        "- `kfold_resumo.csv`: media, desvio-padrao, minimo e maximo entre folds.",
        "- `kfold_folds.md` e `kfold_resumo.md`: versoes rapidas para leitura.",
        "",
        "Artefatos pesados de cada fold ficam nos `output_dir` registrados em `kfold_folds.csv`.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"K-fold article-ready summary saved to: {out.resolve()}")


def cmd_kfold(args) -> None:
    base = load_config(args.config)
    folds = int(args.folds or base.kfold_splits)
    model_name = str(base.kfold_model).lower()
    epochs = int(args.epochs or base.kfold_epochs)
    patience = int(args.patience or base.kfold_patience)
    output_dir = args.output or Path("results") / f"kfold_box{base.box_size_px}_yolo11{model_name}"

    print("=" * 88)
    print(
        f"K-fold: YOLO11{model_name} | box={base.box_size_px}px | folds={folds} | "
        f"epochs={epochs} | patience={patience} | threshold=internal-val"
    )
    print("=" * 88)
    if args.dry_run:
        for fold in range(1, folds + 1):
            fold_config = kfold_model_config(base, model_name, fold, folds, epochs, patience)
            prepare_kfold_dataset(fold_config, fold=fold, n_splits=folds, force=False)
        print("Dry run complete: folds prepared, training not started.")
        return

    rows = []
    start = time.time()
    for fold in range(1, folds + 1):
        print("=" * 88)
        print(f"K-fold training/evaluation: fold {fold}/{folds}")
        print("=" * 88)
        fold_config = kfold_model_config(base, model_name, fold, folds, epochs, patience)
        prepare_kfold_dataset(fold_config, fold=fold, n_splits=folds, force=args.force)
        train_detector(fold_config, force=args.force)
        predict_split(fold_config, "val", force=args.force)
        evaluate_detector(fold_config, "val", force=args.force)
        selected_threshold = select_operating_threshold(fold_config)
        predict_split(fold_config, "test", force=args.force)
        evaluate_detector(fold_config, "test", force=args.force)
        rows.append(fold_operating_row(fold_config, fold, model_name, selected_threshold=selected_threshold, split="test"))

    write_kfold_summary(rows, output_dir=output_dir)
    print(pd.DataFrame(rows).sort_values("fold").to_string(index=False))
    print(f"K-fold time: {(time.time() - start) / 60:.1f} min")


def cmd_status(args) -> None:
    config = load_config(args.config)
    artifacts = [
        ("kfold_summary_csv", Path("results") / f"kfold_box{config.box_size_px}_yolo11{config.kfold_model}" / "kfold_resumo.csv"),
        ("kfold_folds_csv", Path("results") / f"kfold_box{config.box_size_px}_yolo11{config.kfold_model}" / "kfold_folds.csv"),
    ]
    for fold in range(1, config.kfold_splits + 1):
        fold_config = kfold_model_config(
            config,
            config.kfold_model,
            fold,
            config.kfold_splits,
            config.kfold_epochs,
            config.kfold_patience,
        )
        artifacts.extend(
            [
                (f"fold_{fold:02d}_data_yaml", fold_config.data_yaml),
                (f"fold_{fold:02d}_split", fold_config.split_summary_csv),
                (f"fold_{fold:02d}_best", fold_config.best_weights),
                (f"fold_{fold:02d}_threshold", fold_config.metrics_dir / "limiar_operacional.json"),
                (f"fold_{fold:02d}_eval_test", fold_config.metrics_dir / "avaliacao_test_limiares.csv"),
            ]
        )
    for name, path in artifacts:
        if path.exists():
            size_mb = path.stat().st_size / 1024**2
            print(f"[ok]       {name:18s} {path} ({size_mb:.2f} MB)")
        else:
            print(f"[missing]  {name:18s} {path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRIC Cervix single-class cell detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "check": cmd_check,
        "status": cmd_status,
        "download-data": cmd_download_data,
        "kfold": cmd_kfold,
    }
    for name, fn in commands.items():
        command = sub.add_parser(name)
        add_common(command)
        if name == "kfold":
            command.add_argument("--folds", type=int, default=None)
            command.add_argument("--epochs", type=int, default=None)
            command.add_argument("--patience", type=int, default=None)
            command.add_argument("--output", default=None)
            command.add_argument("--dry-run", action="store_true")
        command.set_defaults(func=fn)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
