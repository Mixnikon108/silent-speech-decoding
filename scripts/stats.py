#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests estadísticos pareados (preproc P1 vs raw P0) por experimento y modelo.

Lee los ficheros ``all_results.jsonl`` de cada experimento, empareja los
resultados por semilla y calcula t-test pareado, Wilcoxon, sign test, tamaño
de efecto (Cohen's dz) e intervalos de confianza, con corrección FDR
(Benjamini-Hochberg). Exporta un CSV resumen y figuras por cada par exp/modelo.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, binomtest, t as t_dist

# ---------------------------
# Utilidades
# ---------------------------

def detect_variant(dataset_file: str) -> str:
    """Devuelve 'raw' si contiene _P0_, 'preproc' si contiene _P1_, si no None."""
    if "_P0_" in dataset_file:
        return "raw"
    if "_P1_" in dataset_file:
        return "preproc"
    return None

def cohen_dz(paired_diff: np.ndarray) -> float:
    """Cohen's dz para medidas repetidas: mean(diff)/std(diff, ddof=1)"""
    d = np.asarray(paired_diff)
    sd = d.std(ddof=1)
    if sd == 0:
        return np.nan
    return d.mean() / sd

def mean_diff_ci(paired_diff: np.ndarray, alpha=0.05):
    """IC para la media de la diferencia con t de Student (pares)."""
    n = len(paired_diff)
    if n < 2:
        return (np.nan, np.nan)
    md = paired_diff.mean()
    sd = paired_diff.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = t_dist.ppf(1 - alpha/2, df=n-1)
    return (md - tcrit * se, md + tcrit * se)

def sign_test_pvalue(diffs: np.ndarray):
    """
    Sign test exacto (binomial) ignorando ceros.
    H0: prob(diff>0)=0.5 ; prueba bilateral.
    """
    signs = diffs[np.nonzero(diffs)]  # ignora difs == 0
    n = len(signs)
    if n == 0:
        return np.nan
    k_pos = (signs > 0).sum()
    # binomtest es bilateral por defecto (desde SciPy 1.7+)
    return binomtest(k_pos, n=n, p=0.5, alternative='two-sided').pvalue

def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """FDR BH sobre una serie (mantiene el índice)."""
    s = pvals.copy().astype(float)
    mask = s.notna()
    m = mask.sum()
    if m == 0:
        return pd.Series(np.nan, index=s.index)
    ranks = s[mask].rank(method='first')
    adj = (s[mask] * m / ranks).clip(upper=1.0)
    # Hacer proced. monótono (desde el final)
    adj_sorted = adj.sort_values(ascending=False).cummin()
    adj_final = pd.Series(index=s.index, dtype=float)
    adj_final.loc[adj_sorted.index] = adj_sorted
    return adj_final

def nice_model_name(model):
    return str(model)

# ---------------------------
# Parsing de resultados
# ---------------------------

def load_results(base_dir: Path, experiments=('sub2','sub5','sub9','all')) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        fpath = base_dir / exp / "all_results.jsonl"
        if not fpath.exists():
            print(f"[Aviso] No existe: {fpath}")
            continue
        with fpath.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("[RESULT_JSON]"):
                    line = line.replace("[RESULT_JSON] ", "", 1)
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dataset_file = rec.get("dataset_file", "")
                variant = detect_variant(dataset_file)
                if variant is None:
                    continue
                rows.append({
                    "exp": exp,
                    "model": rec.get("model"),
                    "seed": rec.get("seed"),
                    "variant": variant,
                    "acc": rec.get("test_accuracy"),
                    "run_id": rec.get("run_id"),
                    "dataset_file": dataset_file,
                })
    df = pd.DataFrame(rows)
    return df

# ---------------------------
# Estadísticos por exp/modelo
# ---------------------------

def compute_pairwise_stats(df_exp_model: pd.DataFrame):
    """
    Espera filas con columnas: seed, variant in {'raw','preproc'}, acc
    Devuelve dict con métricas y arrays alineados.
    """
    # pares solo donde hay ambas variantes para la MISMA seed
    pivot = df_exp_model.pivot_table(index="seed", columns="variant", values="acc", aggfunc='mean')
    pivot = pivot.dropna(subset=["raw", "preproc"], how="any")
    seeds_paired = pivot.index.tolist()
    if len(pivot) < 2:  # Wilcoxon y t-test necesitan al menos 2 pares (técnicamente t con n=2 es posible, pero muy débil)
        return {
            "n_pairs": len(pivot),
            "raw_mean": df_exp_model[df_exp_model["variant"]=="raw"]["acc"].mean(),
            "preproc_mean": df_exp_model[df_exp_model["variant"]=="preproc"]["acc"].mean(),
            "mdiff": np.nan,
            "mdiff_ci_low": np.nan,
            "mdiff_ci_high": np.nan,
            "t_stat": np.nan, "t_p": np.nan,
            "w_stat": np.nan, "w_p": np.nan,
            "sign_p": np.nan,
            "cohen_dz": np.nan,
            "seeds_paired": seeds_paired,
        }
    raw = pivot["raw"].to_numpy()
    pre = pivot["preproc"].to_numpy()
    diffs = pre - raw  # DIFERENCIA DEFINIDA COMO (preproc - raw)

    # t-test pareado
    t_stat, t_p = ttest_rel(pre, raw)

    # Wilcoxon (manejo de casos degenerados)
    try:
        w_stat, w_p = wilcoxon(pre, raw, zero_method='wilcox', correction=False, alternative='two-sided', mode='exact')
    except ValueError:
        # p.ej., todas las diferencias 0
        w_stat, w_p = (np.nan, np.nan)

    # Sign test exacto (binomial)
    sign_p = sign_test_pvalue(diffs)

    # Efecto y CI
    dz = cohen_dz(diffs)
    ci_low, ci_high = mean_diff_ci(diffs)

    return {
        "n_pairs": len(diffs),
        "raw_mean": raw.mean(),
        "preproc_mean": pre.mean(),
        "mdiff": diffs.mean(),
        "mdiff_ci_low": ci_low,
        "mdiff_ci_high": ci_high,
        "t_stat": t_stat, "t_p": t_p,
        "w_stat": w_stat, "w_p": w_p,
        "sign_p": sign_p,
        "cohen_dz": dz,
        "seeds_paired": seeds_paired,
    }

# ---------------------------
# Plots
# ---------------------------

def plot_paired_and_box(exp, model, raw_vals, pre_vals, seeds, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Figura 1: puntos apareados y líneas
    plt.figure(figsize=(6,5))
    x1, x2 = np.full_like(raw_vals, 1, dtype=float), np.full_like(pre_vals, 2, dtype=float)
    for i in range(len(raw_vals)):
        plt.plot([1,2], [raw_vals[i], pre_vals[i]], marker='o')
    plt.xticks([1,2], ['raw (P0)', 'preproc (P1)'])
    plt.ylabel('Test Accuracy')
    plt.title(f'{exp} | {model} — Paired Acc por seed')
    for i, s in enumerate(seeds):
        plt.text(1.02, raw_vals[i], f'seed {s}', fontsize=8)
    plt.tight_layout()
    f1 = outdir / f"{exp}__{model}__paired.png"
    plt.savefig(f1, dpi=200)
    plt.close()

    # Figura 2: boxplot simple
    plt.figure(figsize=(5,5))
    plt.boxplot([raw_vals, pre_vals], labels=['raw (P0)','preproc (P1)'])
    plt.ylabel('Test Accuracy')
    plt.title(f'{exp} | {model} — Boxplot Acc')
    plt.tight_layout()
    f2 = outdir / f"{exp}__{model}__box.png"
    plt.savefig(f2, dpi=200)
    plt.close()

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Tests estadísticos: preproc (P1) vs raw (P0) por experimento y modelo.")
    parser.add_argument("--base_dir", type=str,
                        default="/path/to/project/EXPREAL",
                        help="Carpeta que contiene sub2, sub5, sub9 y all")
    parser.add_argument("--experiments", type=str, nargs="*", default=["sub2","sub5","sub9","all"],
                        help="Lista de experimentos a incluir")
    parser.add_argument("--out_dir", type=str, default="./stats_out",
                        help="Carpeta de salida para CSV y figuras")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(base_dir, experiments=args.experiments)
    if df.empty:
        print("[Error] No se leyeron resultados. Revisa la ruta.")
        return

    # Guardar datos crudos
    df.to_csv(out_dir / "acc_data_raw.csv", index=False)

    # Calcular estadísticas por (exp, model)
    summaries = []
    plots_dir = out_dir / "plots"

    for (exp, model), df_em in df.groupby(["exp","model"], sort=False):
        stats = compute_pairwise_stats(df_em)
        stats_row = {
            "exp": exp,
            "model": model,
            "n_pairs": stats["n_pairs"],
            "raw_mean": stats["raw_mean"],
            "preproc_mean": stats["preproc_mean"],
            "mdiff_pre_minus_raw": stats["mdiff"],
            "mdiff_ci_low": stats["mdiff_ci_low"],
            "mdiff_ci_high": stats["mdiff_ci_high"],
            "t_stat": stats["t_stat"], "t_p": stats["t_p"],
            "w_stat": stats["w_stat"], "w_p": stats["w_p"],
            "sign_p": stats["sign_p"],
            "cohen_dz": stats["cohen_dz"],
            "seeds_paired": ";".join(map(str, stats["seeds_paired"])) if isinstance(stats["seeds_paired"], list) else "",
        }
        summaries.append(stats_row)

        # Plots si tenemos pares
        pivot = df_em.pivot_table(index="seed", columns="variant", values="acc", aggfunc='mean').dropna(subset=["raw","preproc"], how="any")
        if len(pivot) >= 2:
            raw_vals = pivot["raw"].to_numpy()
            pre_vals = pivot["preproc"].to_numpy()
            seeds = list(pivot.index)
            safe_model = nice_model_name(model).replace("/", "_")
            plot_paired_and_box(exp, safe_model, raw_vals, pre_vals, seeds, plots_dir)

    summary_df = pd.DataFrame(summaries)

    # Ajuste FDR por tipo de p-valor
    if not summary_df.empty:
        summary_df["t_p_fdr"] = benjamini_hochberg(summary_df["t_p"])
        summary_df["w_p_fdr"] = benjamini_hochberg(summary_df["w_p"])
        summary_df["sign_p_fdr"] = benjamini_hochberg(summary_df["sign_p"])

    # Ordenar columnas
    cols = [
        "exp","model","n_pairs",
        "raw_mean","preproc_mean","mdiff_pre_minus_raw","mdiff_ci_low","mdiff_ci_high",
        "cohen_dz",
        "t_stat","t_p","t_p_fdr",
        "w_stat","w_p","w_p_fdr",
        "sign_p","sign_p_fdr",
        "seeds_paired"
    ]
    summary_df = summary_df.reindex(columns=cols)
    summary_path = out_dir / "stats_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Imprimir resumen bonito por consola
    pd.set_option("display.precision", 4)
    print("\n===== RESUMEN (preproc - raw) por experimento y modelo =====\n")
    print(summary_df.fillna("NA").to_string(index=False))
    print(f"\n[OK] Guardado CSV con datos crudos en: {out_dir/'acc_data_raw.csv'}")
    print(f"[OK] Guardado CSV resumen en: {summary_path}")
    print(f"[OK] Figuras en: {plots_dir}")

if __name__ == "__main__":
    main()
