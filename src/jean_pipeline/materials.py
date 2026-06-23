from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import Config
from .data import load_metadata


PT_NAMES = {
    "precisao": "Precisao",
    "revocacao": "Revocacao",
    "f1": "F1",
    "mae_contagem": "MAE contagem",
    "rmse_contagem": "RMSE contagem",
    "bias_contagem": "Vies contagem",
    "corr_contagem": "Correlacao contagem",
    "ap": "AP",
}


def fmt(value) -> str:
    try:
        return f"{float(value):.4f}".replace(".", ",")
    except Exception:
        return str(value)


def to_markdown(table: pd.DataFrame, index: bool = False) -> str:
    data = table.copy()
    if index:
        data = data.reset_index()
    lines = [
        "| " + " | ".join(map(str, data.columns)) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]
    for row in data.to_numpy():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines) + "\n"


def write_table(table: pd.DataFrame, target: Path, index: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target.with_suffix(".csv"), index=index, encoding="utf-8")
    target.with_suffix(".md").write_text(to_markdown(table, index=index), encoding="utf-8")
    target.with_suffix(".tex").write_text(table.to_latex(index=index, escape=False), encoding="utf-8")


def copy_if_exists(source: Path, target: Path) -> bool:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return True
    return False


def make_materials(config: Config, output_dir: str | Path = "materiais_artigo") -> None:
    out = Path(output_dir)
    figures = out / "figuras"
    tables = out / "tabelas"
    metrics = out / "metricas"
    raw = out / "dados_origem"
    for path in [figures, tables, metrics, raw]:
        path.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(config)
    build_dataset_tables(config, metadata, tables)
    build_result_tables(config, tables)
    build_extra_figures(config, metadata, figures)
    copy_artifacts(config, figures, raw)
    write_summary(config, metrics)
    write_readme(out)
    print(f"Article materials generated at: {out.resolve()}")


def build_dataset_tables(config: Config, metadata: pd.DataFrame, tables: Path) -> None:
    split = pd.read_csv(config.split_summary_csv) if config.split_summary_csv.exists() else pd.DataFrame()
    if not split.empty:
        write_table(split, tables / "tabela_particoes")

    composition = (
        metadata["classe_bethesda"]
        .replace({"Negative for intraepithelial lesion": "NILM"})
        .value_counts()
        .rename_axis("Categoria Bethesda")
        .reset_index(name="Celulas")
    )
    composition["Percentual"] = (100 * composition["Celulas"] / composition["Celulas"].sum()).round(2)
    write_table(composition, tables / "tabela_composicao_bethesda_original")

    target = pd.DataFrame(
        [
            ("Tarefa", "Deteccao de todas as celulas, classe unica cell"),
            ("Anotacao original", "Ponto nuclear x/y por celula"),
            ("Pseudo-bounding box principal", f"{config.box_size_px} x {config.box_size_px} px"),
            ("Tamanhos para ablacao", ", ".join(map(str, config.box_size_candidates))),
            ("Split", "Por imagem, sem compartilhar uma imagem entre treino/validacao/teste"),
            ("Base", "CRIC Cervix, 400 imagens e 11534 celulas anotadas"),
        ],
        columns=["Item", "Valor"],
    )
    write_table(target, tables / "tabela_definicao_alvo_deteccao")


def build_result_tables(config: Config, tables: Path) -> None:
    for split in ["val", "test"]:
        path = config.metrics_dir / f"avaliacao_{split}_limiares.csv"
        if path.exists():
            table = pd.read_csv(path)
            table = table.rename(columns=PT_NAMES)
            write_table(table, tables / f"tabela_avaliacao_{split}_limiares")
        path = config.metrics_dir / f"map_{split}.csv"
        if path.exists():
            table = pd.read_csv(path).rename(columns=PT_NAMES)
            write_table(table, tables / f"tabela_map_{split}")

    final_path = config.metrics_dir / "metricas_operacionais_teste.csv"
    if final_path.exists():
        final = pd.read_csv(final_path).rename(columns=PT_NAMES)
        write_table(final, tables / "tabela_metricas_operacionais_teste")

    boot_path = config.metrics_dir / "intervalos_confianca_bootstrap.csv"
    if boot_path.exists():
        boot = pd.read_csv(boot_path).rename(columns={"metrica": "Metrica", "valor": "Valor", "ic95": "IC95"})
        boot["Metrica"] = boot["Metrica"].replace(PT_NAMES)
        write_table(boot, tables / "tabela_ic95_bootstrap")

    sweep_path = config.metrics_dir / "bbox_sweep_summary.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path).rename(columns=PT_NAMES)
        write_table(sweep, tables / "tabela_ablacao_tamanho_bbox")

    compare_path = config.metrics_dir / "model_compare_summary.csv"
    if compare_path.exists():
        compare = pd.read_csv(compare_path).rename(columns=PT_NAMES)
        write_table(compare, tables / "tabela_comparacao_modelos_yolo")


def build_extra_figures(config: Config, metadata: pd.DataFrame, figures: Path) -> None:
    counts = metadata.groupby("particao").size().reindex(["train", "val", "test"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(counts.index, counts.values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Cells")
    ax.set_title("CRIC cell annotations by split")
    for i, value in enumerate(counts.values):
        ax.text(i, value + max(counts.values) * 0.015, str(int(value)), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "fig_particoes_celulas.png", dpi=220)
    plt.close(fig)

    composition = metadata["classe_bethesda"].replace({"Negative for intraepithelial lesion": "NILM"}).value_counts()
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(composition.index, composition.values, color="#4C78A8")
    ax.set_ylabel("Cells")
    ax.set_title("Original Bethesda distribution used only for auditing")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "fig_composicao_bethesda_original.png", dpi=220)
    plt.close(fig)


def copy_artifacts(config: Config, figures: Path, raw: Path) -> None:
    figure_map = {
        "auditoria_boxes_pseudo_gt.png": "fig_auditoria_boxes_pseudo_gt.png",
        "f1_por_limiar_conf_iou.png": "fig_f1_por_limiar_conf_iou.png",
        "contagem_gt_vs_pred_teste.png": "fig_contagem_gt_vs_pred_teste.png",
    }
    for source_name, target_name in figure_map.items():
        copy_if_exists(config.figures_dir / source_name, figures / target_name)

    files = [
        config.metadata_csv,
        config.split_summary_csv,
        config.metrics_dir / "composicao_bethesda_original.csv",
        config.metrics_dir / "contagem_celulas_por_imagem_gt.csv",
        config.metrics_dir / "avaliacao_val_limiares.csv",
        config.metrics_dir / "avaliacao_test_limiares.csv",
        config.metrics_dir / "map_val.csv",
        config.metrics_dir / "map_test.csv",
        config.metrics_dir / "metricas_operacionais_teste.csv",
        config.metrics_dir / "intervalos_confianca_bootstrap.csv",
        config.metrics_dir / "contagem_por_imagem_teste.csv",
        config.metrics_dir / "limiar_operacional.json",
        config.metrics_dir / "treino_detector_config.json",
        config.metrics_dir / "bbox_sweep_summary.csv",
        config.metrics_dir / "model_compare_summary.csv",
    ]
    for source in files:
        if source.exists():
            copy_if_exists(source, raw / source.name)


def write_summary(config: Config, metrics: Path) -> None:
    summary = {}
    for key, source in {
        "operacional_teste": config.metrics_dir / "metricas_operacionais_teste.csv",
        "map_test": config.metrics_dir / "map_test.csv",
        "ic95": config.metrics_dir / "intervalos_confianca_bootstrap.csv",
        "limiar_operacional": config.metrics_dir / "limiar_operacional.json",
        "bbox_sweep": config.metrics_dir / "bbox_sweep_summary.csv",
        "comparacao_modelos": config.metrics_dir / "model_compare_summary.csv",
    }.items():
        if source.exists():
            if source.suffix == ".json":
                summary[key] = json.loads(source.read_text(encoding="utf-8"))
            else:
                summary[key] = pd.read_csv(source).to_dict(orient="records")
    (metrics / "resumo_metricas.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Resumo das metricas", ""]
    final_path = config.metrics_dir / "metricas_operacionais_teste.csv"
    if final_path.exists():
        final = pd.read_csv(final_path).iloc[0]
        lines.append(f"- Teste IoU={final['iou']:.2f}, conf={final['conf']:.2f}: F1 {final['f1']:.4f}, precisao {final['precisao']:.4f}, revocacao {final['revocacao']:.4f}.")
        lines.append(f"- Contagem por imagem: MAE {final['mae_contagem']:.2f}, bias {final['bias_contagem']:.2f}.")
    map_path = config.metrics_dir / "map_test.csv"
    if map_path.exists():
        map_df = pd.read_csv(map_path)
        for row in map_df.itertuples(index=False):
            lines.append(f"- AP teste IoU={row.iou:.2f}: {row.ap:.4f}.")
    (metrics / "resumo_metricas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(out: Path) -> None:
    lines = [
        "# Materiais para o artigo Jean",
        "",
        "Pasta gerada automaticamente a partir de `outputs_detection`.",
        "",
        "## Figuras",
    ]
    lines.extend(f"- `figuras/{path.name}`" for path in sorted((out / "figuras").glob("*.png")))
    lines += ["", "## Tabelas"]
    lines.extend(f"- `tabelas/{path.name}` (+ .md e .tex)" for path in sorted((out / "tabelas").glob("*.csv")))
    lines += [
        "",
        "## Dados de origem",
        "",
        "Os CSVs em `dados_origem` preservam metricas brutas, limiares, contagens por imagem e configuracoes de treino.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
