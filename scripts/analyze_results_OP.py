#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis de resultados por dataset con intervalos de confianza al 95%.

Carga un fichero ``all_results.jsonl``, resume la métrica elegida por
(modelo, dataset), calcula deltas pareados por semilla entre datasets
(P0 vs P1) y genera una figura de barras con IC95 en formato PNG y PDF.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================= utils =============================

def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "[RESULT_JSON]" in s:
                s = s[s.index("{"):]
            if "{" not in s or "}" not in s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows

def infer_dataset_tag(dataset_file: Optional[str]) -> str:
    if not dataset_file:
        return "UNK"
    stem = Path(dataset_file).stem
    if "_P0" in stem or stem.endswith("P0"):
        return "P0"
    if "_P1" in stem or stem.endswith("P1"):
        return "P1"
    for tok in stem.split("_"):
        if tok.startswith("P") and len(tok) == 2 and tok[1].isdigit():
            return tok
    return "ALL"

def ensure_model_column(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns and "model_name" in df.columns:
        df["model"] = df["model_name"]
    return df

def ci95_halfwidth_from_series(x: pd.Series, use_t: bool = True) -> float:
    x = x.dropna()
    n = len(x)
    if n <= 1:
        return np.nan
    sem = x.std(ddof=1) / np.sqrt(n)
    if not use_t:
        return 1.96 * sem
    try:
        from scipy.stats import t
        tcrit = t.ppf(0.975, df=n - 1)
    except Exception:
        tcrit = 1.96
    return float(tcrit * sem)

def latexish_matplotlib_style():
    # Estética tipo paper sin LaTeX duro (evita dvipng)
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.linewidth": 0.25,
    })

# ---------- pretty helpers ----------

_MODEL_MAP = {
    "DiffE_AT": "Diff-E AT",
    "cnn": "CNN",
    "DiffE_mod": "Diff-E Base",
    "shallowconvnet": "ShallowConvNet",
    "eegnet": "EEGNet",
    "deepconvnet": "DeepConvNet",
    "DiffE_TF": "Diff-E TF",
    "mlp": "MLP",
}

def prettify_model(name: str) -> str:
    return _MODEL_MAP[name]
    

def prettify_metric_label(metric: str) -> str:
    # "test_accuracy" -> "Test accuracy"
    words = metric.replace("_", " ").split()
    if not words:
        return metric
    words[0] = words[0].capitalize()
    for i in range(1, len(words)):
        words[i] = words[i].lower()
    return " ".join(words)

_DATASET_LABEL = {
    "P0": "Raw dataset",
    "P1": "Preprocessed",
}

# ============================ core ==============================

def summarize_by_model_dataset(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    gcols = ["model", "dataset_tag"]
    out = (
        df.groupby(gcols)
          .agg(n=(metric, "count"),
               mean=(metric, "mean"),
               sd=(metric, lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan))
          .reset_index()
    )
    ci = []
    for _, row in out.iterrows():
        mask = (df["model"] == row["model"]) & (df["dataset_tag"] == row["dataset_tag"])
        ci.append(ci95_halfwidth_from_series(df.loc[mask, metric]))
    out["ci95"] = ci
    out["summary"] = out["mean"].map(lambda v: f"{v:.3f}") + " ± " + out["ci95"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "NA")
    out["model_display"] = out["model"].astype(str).map(prettify_model)
    return out

def paired_delta_by_model(df: pd.DataFrame, metric: str,
                          ds_a: str = "P0", ds_b: str = "P1") -> pd.DataFrame:
    """
    Delta emparejado por seed: (ds_a - ds_b). Por defecto: P0 - P1.
    """
    results = []
    for model, sub in df.groupby("model"):
        a = sub[sub["dataset_tag"] == ds_a].dropna(subset=["seed", metric])
        b = sub[sub["dataset_tag"] == ds_b].dropna(subset=["seed", metric])
        if a.empty or b.empty:
            continue
        common = sorted(set(a["seed"]).intersection(set(b["seed"])))
        if len(common) == 0:
            continue
        a_map = a.set_index("seed")[metric]
        b_map = b.set_index("seed")[metric]
        diffs = pd.Series([a_map[s] - b_map[s] for s in common], index=common, dtype=float)  # <-- P0 - P1
        n = len(diffs)
        mean_delta = float(diffs.mean())
        sd_delta = float(diffs.std(ddof=1)) if n > 1 else np.nan
        ci_hw = ci95_halfwidth_from_series(diffs)
        results.append({
            "model": model,
            "model_display": prettify_model(model),
            "dataset_A": ds_a,
            "dataset_B": ds_b,
            "n_paired": n,
            "mean_delta": mean_delta,
            "sd_delta": sd_delta,
            "ci95_delta": ci_hw,
            "summary": f"{mean_delta:+.3f} ± {ci_hw:.3f}" if pd.notna(ci_hw) else f"{mean_delta:+.3f} ± NA"
        })
    cols = ["model","model_display","dataset_A","dataset_B","n_paired","mean_delta","sd_delta","ci95_delta","summary"]
    return pd.DataFrame(results, columns=cols).sort_values("model_display").reset_index(drop=True)

def plot_bars_by_dataset(summary_df: pd.DataFrame,
                         metric_label: str,
                         out_png: Path,
                         out_pdf: Path,
                         title: Optional[str] = None,
                         datasets_order: Optional[Tuple[str, str]] = ("P0", "P1"),
                         baseline: Optional[float] = 0.2):
    """
    Barras por modelo separadas por dataset, con IC95, estética paper.
    - metric_label: texto bonito para el eje Y (p.ej., "Test accuracy")
    - baseline: dibuja línea horizontal si no es None
    """
    latexish_matplotlib_style()

    # Dataset labels y colores (pastel): P0 verde, P1 naranja
    pastel_colors = {"P0": "#93C47D", "P1": "#F6B26B"}  # verde pastel, naranja pastel

    # Si no se pasa orden, usa (P0,P1) si existen
    if datasets_order is None:
        datasets = [d for d in ("P0","P1") if d in summary_df["dataset_tag"].unique().tolist()]
        if not datasets:
            datasets = sorted(summary_df["dataset_tag"].unique().tolist())
    else:
        datasets = [d for d in datasets_order if d in summary_df["dataset_tag"].unique().tolist()]

    # Pivot con nombres bonitos
    tmp = summary_df.copy()
    tmp["model_display"] = tmp["model_display"].astype(str)
    piv_mean = tmp.pivot_table(index="model_display", columns="dataset_tag", values="mean", aggfunc="mean")
    piv_ci   = tmp.pivot_table(index="model_display", columns="dataset_tag", values="ci95", aggfunc="mean")

    # Ordenar modelos por la puntuación en Raw (P0) descendente
    if "P0" in piv_mean.columns:
        order = piv_mean["P0"].sort_values(ascending=False).index.tolist()
        # si hay modelos sin P0, añádelos al final en orden alfabético
        rest = [m for m in piv_mean.index if m not in order]
        order += sorted(rest)
    else:
        # fallback: por media global
        order = piv_mean.mean(axis=1).sort_values(ascending=False).index.tolist()

    piv_mean = piv_mean.reindex(index=order, columns=datasets)
    piv_ci   = piv_ci.reindex(index=order, columns=datasets)

    models_display = piv_mean.index.tolist()
    n_models = len(models_display)
    n_ds = len(datasets)
    width = 0.38 if n_ds == 2 else 0.28
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(7.6, 0.9 * n_models + 4), 4.0), constrained_layout=True)

    # Línea baseline
    baseline_handle = None
    if baseline is not None:
        baseline_handle = ax.axhline(baseline, linestyle="--", linewidth=0.9, color="#7f7f7f", alpha=0.7, label=f"Baseline {baseline:.1f}")

    # Barras + errorbars
    handles, labels = [], []
    for j, ds in enumerate(datasets):
        offs = (j - (n_ds - 1) / 2) * (width * 1.2)
        y = piv_mean[ds].values
        yerr = piv_ci[ds].values
        color = pastel_colors.get(ds, "#93C47D")  # default verde pastel
        h = ax.bar(x + offs, y, width=width,
                   label=_DATASET_LABEL.get(ds, ds),
                   linewidth=0.7, edgecolor="black",
                   color=color)
        ax.errorbar(x + offs, y, yerr=yerr, fmt="none",
                    elinewidth=0.7, capsize=2, capthick=0.7, ecolor="black")
        handles.append(h)
        labels.append(_DATASET_LABEL.get(ds, ds))

    ax.set_xticks(x)
    ax.set_xticklabels(models_display, rotation=30, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} by model and dataset")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Leyenda: datasets (+ baseline si está)
    legend_handles = [h for h in handles]
    legend_labels  = labels[:]
    if baseline_handle is not None:
        legend_handles.append(baseline_handle)
        legend_labels.append("Baseline 0.2")
    ax.legend(legend_handles, legend_labels, frameon=False, ncol=min(n_ds+1, 3), loc="upper right")

    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)

# ============================ main ==============================

def main():
    ap = argparse.ArgumentParser(description="Analyze results per dataset (95% CI) and compare datasets per model.")
    ap.add_argument("--jsonl", required=True, type=str, help="Path to all_results.jsonl")
    ap.add_argument("--metric", type=str, default="test_accuracy", help="Metric key (e.g., test_accuracy, test_macro_f1)")
    ap.add_argument("--out-dir", type=str, default="results/analysis", help="Output folder")
    ap.add_argument("--title", type=str, default=None, help="Optional figure title (English)")
    ap.add_argument("--ds-a", type=str, default="P0", help="Dataset A (baseline) for delta")
    ap.add_argument("--ds-b", type=str, default="P1", help="Dataset B (comparison) for delta")
    ap.add_argument("--baseline", type=float, default=0.2, help="Horizontal baseline line (None to disable)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(jsonl_path)
    if not rows:
        raise SystemExit(f"Could not read results from: {jsonl_path}")

    df = pd.DataFrame(rows)
    df = ensure_model_column(df)

    if "dataset_file" not in df.columns:
        df["dataset_file"] = ""
    df["dataset_tag"] = df["dataset_file"].apply(infer_dataset_tag)

    if "seed" not in df.columns:
        df["seed"] = np.arange(len(df))

    metric = args.metric
    if metric not in df.columns:
        raise SystemExit(f"Metric '{metric}' not found. Available columns: {sorted(df.columns)}")

    # 1) CSV con columnas relevantes
    keep_cols = ["model", "dataset_tag", "seed", metric]
    full_csv = out_dir / "results_per_dataset.csv"
    df[keep_cols].to_csv(full_csv, index=False)
    print(f"✅ Row-wise results → {full_csv}")

    # 2) Resumen por (modelo, dataset) con nombres bonitos
    summ = summarize_by_model_dataset(df, metric)
    # ordenar para CSV por nombre bonito y dataset
    summ_csv = (summ.sort_values(["model_display","dataset_tag"])
                    .reset_index(drop=True))
    ci_csv = out_dir / f"summary_{metric}_by_model_dataset.csv"
    summ_csv.to_csv(ci_csv, index=False)
    print(f"✅ Summary by model+dataset (95% CI) → {ci_csv}")

    # 3) Deltas emparejados (P0 - P1 por defecto)
    deltas = paired_delta_by_model(df, metric, ds_a=args.ds_a, ds_b=args.ds_b)
    delta_csv = out_dir / f"deltas_{metric}_{args.ds_a}_minus_{args.ds_b}.csv"
    deltas.to_csv(delta_csv, index=False)
    print(f"✅ Paired delta ({args.ds_a} - {args.ds_b}) with 95% CI → {delta_csv}")

    # 4) Figura con mejoras solicitadas
    metric_label = prettify_metric_label(metric)  # "Test accuracy"
    bars_png = out_dir / f"bars_{metric}_by_dataset.png"
    bars_pdf = out_dir / f"bars_{metric}_by_dataset.pdf"
    plot_bars_by_dataset(
        summ,
        metric_label=metric_label,
        out_png=bars_png,
        out_pdf=bars_pdf,
        title=args.title,
        datasets_order=(args.ds_a, args.ds_b) if args.ds_a != args.ds_b else None,
        baseline=args.baseline
    )
    print(f"✅ Figure saved at: {bars_png} and {bars_pdf}")

    # 5) Consola
    print("\n--- 95% CI by model & dataset ---")
    print(summ_csv[["model_display", "dataset_tag", "n", "summary"]]
          .rename(columns={"model_display":"model"}).to_string(index=False))

    if not deltas.empty:
        print("\n--- Paired delta ({} - {}) by model ---".format(args.ds_a, args.ds_b))
        print(deltas[["model_display", "n_paired", "summary"]]
              .rename(columns={"model_display":"model"}).to_string(index=False))
    else:
        print("\n(ℹ️) Could not compute paired deltas: no common seeds between datasets for some/all models.")

if __name__ == "__main__":
    main()