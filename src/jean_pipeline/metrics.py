from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .data import load_metadata


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_boxes_by_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Match unique prediction/GT pairs by descending IoU, as in Ultralytics validation."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.empty((0, 2), dtype=int)
    ious = box_iou_matrix(pred_boxes, gt_boxes)
    pred_idx, gt_idx = np.where(ious >= iou_threshold)
    if len(pred_idx) == 0:
        return np.empty((0, 2), dtype=int)
    matches = np.column_stack((pred_idx, gt_idx, ious[pred_idx, gt_idx]))
    matches = matches[np.argsort(matches[:, 2])[::-1]]
    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    return matches[:, :2].astype(int)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(
            columns=["arquivo_imagem", "conf", "pred_x1", "pred_y1", "pred_x2", "pred_y2"]
        )
    table = pd.read_csv(path)
    if table.empty:
        return table
    return table


def greedy_match(gt: pd.DataFrame, pred: pd.DataFrame, iou_threshold: float, conf_threshold: float) -> dict:
    pred = pred[pred["conf"] >= conf_threshold].sort_values("conf", ascending=False).reset_index(drop=True)
    gt_boxes = gt[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(float)
    pred_boxes = pred[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]].to_numpy(float)
    pairs = match_boxes_by_iou(pred_boxes, gt_boxes, iou_threshold)
    tp = len(pairs)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "gt_count": int(len(gt)),
        "pred_count": int(len(pred)),
        "matches": pairs.tolist(),
    }


def detection_metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precisao": float(precision),
        "revocacao": float(recall),
        "f1": float(f1),
    }


def average_precision(gt: pd.DataFrame, pred: pd.DataFrame, iou_threshold: float) -> float:
    gt_by_image = {name: group for name, group in gt.groupby("arquivo_yolo")}
    pred = pred.copy().reset_index(drop=True)
    pred["_correct"] = False
    for image_name, pred_img in pred.groupby("arquivo_imagem", sort=False):
        gt_img = gt_by_image.get(image_name)
        if gt_img is None or gt_img.empty:
            continue
        pred_boxes = pred_img[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]].to_numpy(float)
        gt_boxes = gt_img[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(float)
        pairs = match_boxes_by_iou(pred_boxes, gt_boxes, iou_threshold)
        if len(pairs):
            pred.loc[pred_img.index[pairs[:, 0]], "_correct"] = True
    pred = pred.sort_values("conf", ascending=False).reset_index(drop=True)
    tp = pred["_correct"].to_numpy(float)
    fp = 1.0 - tp
    total_gt = len(gt)
    if total_gt == 0 or len(pred) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def evaluate_threshold_grid(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    split: str,
    iou_thresholds: list[float],
    conf_grid: list[float],
) -> pd.DataFrame:
    rows = []
    gt_by_image = {name: group for name, group in gt.groupby("arquivo_yolo")}
    pred_by_image = {name: group for name, group in pred.groupby("arquivo_imagem")} if not pred.empty else {}
    image_names = sorted(gt_by_image)
    for iou_thr in iou_thresholds:
        for conf_thr in conf_grid:
            totals = {"tp": 0, "fp": 0, "fn": 0}
            count_rows = []
            for image_name in image_names:
                matched = greedy_match(
                    gt_by_image[image_name],
                    pred_by_image.get(image_name, pd.DataFrame(columns=pred.columns)),
                    iou_thr,
                    conf_thr,
                )
                for key in totals:
                    totals[key] += matched[key]
                count_rows.append(
                    {
                        "arquivo_yolo": image_name,
                        "gt_count": matched["gt_count"],
                        "pred_count": matched["pred_count"],
                    }
                )
            base = detection_metrics_from_counts(**totals)
            counts = pd.DataFrame(count_rows)
            error = counts["pred_count"] - counts["gt_count"]
            base.update(
                {
                    "particao": split,
                    "iou": iou_thr,
                    "conf": conf_thr,
                    "mae_contagem": float(error.abs().mean()),
                    "rmse_contagem": float(np.sqrt(np.mean(np.square(error)))),
                    "bias_contagem": float(error.mean()),
                    "corr_contagem": float(np.corrcoef(counts["gt_count"], counts["pred_count"])[0, 1])
                    if counts["pred_count"].nunique() > 1
                    else 0.0,
                }
            )
            rows.append(base)
    return pd.DataFrame(rows)


def per_image_counts(gt: pd.DataFrame, pred: pd.DataFrame, conf_threshold: float, iou_threshold: float) -> pd.DataFrame:
    gt_by_image = {name: group for name, group in gt.groupby("arquivo_yolo")}
    pred_by_image = {name: group for name, group in pred.groupby("arquivo_imagem")} if not pred.empty else {}
    rows = []
    for image_name in sorted(gt_by_image):
        matched = greedy_match(
            gt_by_image[image_name],
            pred_by_image.get(image_name, pd.DataFrame(columns=pred.columns)),
            iou_threshold,
            conf_threshold,
        )
        rows.append(
            {
                "arquivo_yolo": image_name,
                "gt_count": matched["gt_count"],
                "pred_count": matched["pred_count"],
                "tp": matched["tp"],
                "fp": matched["fp"],
                "fn": matched["fn"],
                "erro_contagem": matched["pred_count"] - matched["gt_count"],
                "erro_abs_contagem": abs(matched["pred_count"] - matched["gt_count"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_detection(per_image: pd.DataFrame, seed: int, replicates: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    values = {key: [] for key in ["precisao", "revocacao", "f1", "mae_contagem", "bias_contagem"]}
    n = len(per_image)
    for _ in range(replicates):
        sample = per_image.iloc[rng.integers(0, n, size=n)]
        metrics = detection_metrics_from_counts(int(sample["tp"].sum()), int(sample["fp"].sum()), int(sample["fn"].sum()))
        error = sample["erro_contagem"]
        values["precisao"].append(metrics["precisao"])
        values["revocacao"].append(metrics["revocacao"])
        values["f1"].append(metrics["f1"])
        values["mae_contagem"].append(float(error.abs().mean()))
        values["bias_contagem"].append(float(error.mean()))
    point = detection_metrics_from_counts(int(per_image["tp"].sum()), int(per_image["fp"].sum()), int(per_image["fn"].sum()))
    point["mae_contagem"] = float(per_image["erro_abs_contagem"].mean())
    point["bias_contagem"] = float(per_image["erro_contagem"].mean())
    for key, samples in values.items():
        lo, hi = np.percentile(samples, [2.5, 97.5])
        rows.append({"metrica": key, "valor": round(point[key], 4), "ic95": f"{point[key]:.3f} [{lo:.3f}-{hi:.3f}]"})
    return pd.DataFrame(rows)


def evaluate_detector(config: Config, split: str = "test", force: bool = False) -> None:
    out_summary = config.metrics_dir / f"avaliacao_{split}_limiares.csv"
    if out_summary.exists() and not force:
        print(f"Using existing evaluation: {out_summary}")
        return
    metadata = load_metadata(config)
    gt = metadata[metadata["particao"] == split].copy()
    pred = load_predictions(config.output_dir / f"predicoes_{split}.csv")
    grid = evaluate_threshold_grid(gt, pred, split, config.iou_thresholds, config.operating_conf_grid)
    grid.to_csv(out_summary, index=False)
    ap_rows = []
    for iou_thr in config.iou_thresholds:
        ap_rows.append({"particao": split, "iou": iou_thr, "ap": average_precision(gt, pred, iou_thr)})
    pd.DataFrame(ap_rows).to_csv(config.metrics_dir / f"map_{split}.csv", index=False)
    print(grid.sort_values(["iou", "f1"], ascending=[True, False]).groupby("iou").head(1).to_string(index=False))


def select_operating_threshold(config: Config) -> dict:
    val_path = config.metrics_dir / "avaliacao_val_limiares.csv"
    if not val_path.exists():
        raise FileNotFoundError("Run eval-val before selecting threshold.")
    val = pd.read_csv(val_path)
    selected = val[val["iou"].eq(0.50)].sort_values(["f1", "revocacao", "precisao"], ascending=False).iloc[0].to_dict()
    payload = {
        "criterio": "maior F1 na validacao em IoU=0.50",
        "conf": float(selected["conf"]),
        "iou": 0.50,
        "metricas_validacao": selected,
    }
    (config.metrics_dir / "limiar_operacional.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def finalize_test_metrics(config: Config, force: bool = False) -> None:
    out = config.metrics_dir / "metricas_operacionais_teste.csv"
    if out.exists() and not force:
        print(f"Using existing final metrics: {out}")
        return
    selected = select_operating_threshold(config)
    metadata = load_metadata(config)
    gt = metadata[metadata["particao"] == "test"].copy()
    pred = load_predictions(config.output_dir / "predicoes_test.csv")
    conf = float(selected["conf"])
    iou = float(selected["iou"])
    per_image = per_image_counts(gt, pred, conf, iou)
    per_image.to_csv(config.metrics_dir / "contagem_por_imagem_teste.csv", index=False)
    boot = bootstrap_detection(per_image, config.seed, config.bootstrap_replicates)
    boot.to_csv(config.metrics_dir / "intervalos_confianca_bootstrap.csv", index=False)
    totals = detection_metrics_from_counts(int(per_image["tp"].sum()), int(per_image["fp"].sum()), int(per_image["fn"].sum()))
    error = per_image["erro_contagem"]
    summary = pd.DataFrame(
        [
            {
                "particao": "test",
                "iou": iou,
                "conf": conf,
                **totals,
                "mae_contagem": float(error.abs().mean()),
                "rmse_contagem": float(np.sqrt(np.mean(np.square(error)))),
                "bias_contagem": float(error.mean()),
                "gt_total": int(per_image["gt_count"].sum()),
                "pred_total": int(per_image["pred_count"].sum()),
                "imagens": int(len(per_image)),
            }
        ]
    )
    summary.to_csv(out, index=False)
    save_metric_figures(config)
    print(summary.to_string(index=False))


def save_metric_figures(config: Config) -> None:
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    test_grid = config.metrics_dir / "avaliacao_test_limiares.csv"
    if test_grid.exists():
        grid = pd.read_csv(test_grid)
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for iou, group in grid.groupby("iou"):
            ax.plot(group["conf"], group["f1"], marker="o", label=f"IoU {iou:.2f}")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("F1")
        ax.set_title("F1 by confidence threshold and IoU")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.figures_dir / "f1_por_limiar_conf_iou.png", dpi=220)
        plt.close(fig)

    counts_path = config.metrics_dir / "contagem_por_imagem_teste.csv"
    if counts_path.exists():
        counts = pd.read_csv(counts_path)
        fig, ax = plt.subplots(figsize=(5.8, 5.6))
        ax.scatter(counts["gt_count"], counts["pred_count"], alpha=0.8, color="#4C78A8")
        limit = max(counts["gt_count"].max(), counts["pred_count"].max()) + 5
        ax.plot([0, limit], [0, limit], color="#222222", linestyle="--", linewidth=1)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Ground-truth cells per image")
        ax.set_ylabel("Detected cells per image")
        ax.set_title("Cell count agreement on test images")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(config.figures_dir / "contagem_gt_vs_pred_teste.png", dpi=220)
        plt.close(fig)
