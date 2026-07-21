#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Informe de ablación de los pasos de preprocesado (BP, N, CAR, ICA, BL).

Parsea las banderas de cada paso desde el nombre del dataset en
``all_results.jsonl`` y genera dos figuras de barras: el impacto de añadir
cada paso al esquema RAW (RAW+P − RAW) y el impacto de retirar cada paso del
esquema FULL (FULL − (FULL−P)).
"""

import json, re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Configuración de rutas/entradas ----------------
jsonl_path = Path("/path/to/project/ABLACION/all_results.jsonl")
out_dir    = Path("/path/to/project/ABLACION")
out_raw    = out_dir / "raw_vs_rawp_bar.pdf"
out_full   = out_dir / "full_vs_fullm_bar.pdf"

metric = "test_accuracy"  # test_accuracy | test_macro_f1 | ...

STEP_NAMES = ["BP", "N", "CAR", "ICA", "BL"]
STEP_LABELS = {
    "BP": "Band-Pass Filter",
    "N": "Notch Filter",
    "CAR": "CAR",
    "ICA": "ICA",
    "BL": "Baseline Removal",
}
FLAGS_REGEX = re.compile(r"BP(?P<BP>[01])_N(?P<N>[01])_CAR(?P<CAR>[01])_ICA(?P<ICA>[01])_BL(?P<BL>[01])")

# ---------------- Estilo y pretty helpers ----------------
def latexish_matplotlib_style():
    # Estética tipo paper sin LaTeX duro
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
    return _MODEL_MAP.get(name, name)

def prettify_metric_label(m: str) -> str:
    words = m.replace("_", " ").split()
    if not words: 
        return m
    words[0] = words[0].capitalize()
    for i in range(1, len(words)):
        words[i] = words[i].lower()
    return " ".join(words)

# ---------------- Parsers ----------------
def extract_flags(path: str):
    m = FLAGS_REGEX.search(path)
    return tuple(int(m.group(name)) for name in STEP_NAMES) if m else None

def scheme_from_flags(flags):
    FULL = (1,1,1,1,1)
    RAW  = (0,0,0,0,0)
    if flags == RAW:
        return "RAW", None
    if flags == FULL:
        return "FULL", None
    s = sum(flags)
    if s == 1:  # RAW+P
        step_idx = flags.index(1)
        return f"RAW+{STEP_NAMES[step_idx]}", STEP_NAMES[step_idx]
    if s == 4:  # FULL-P
        step_idx = flags.index(0)
        return f"FULL-{STEP_NAMES[step_idx]}", STEP_NAMES[step_idx]
    return "OTHER", None

# ---------------- Carga JSONL ----------------
rows = []
with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "{" not in line:
            continue
        payload = json.loads(line[line.index("{"):])
        ds = (payload.get("dataset_file") or "") or (payload.get("exp_dir") or "")
        flags = extract_flags(ds) or extract_flags(payload.get("exp_dir","") or "")
        if not flags:
            continue
        scheme, step = scheme_from_flags(flags)
        rows.append({
            "model": payload.get("model"),
            "seed": payload.get("seed"),
            "scheme": scheme,
            "step": step,
            metric: payload.get(metric)
        })

df = pd.DataFrame(rows).dropna(subset=["model", "seed", metric])
df["seed"] = df["seed"].astype(int)
df[metric] = df[metric].astype(float)

# Orden “bonito” por nombre prettificado
if not df.empty:
    df["model_pretty"] = df["model"].map(prettify_model)
else:
    print("[WARN] No hay datos tras parsear el JSONL.")
    exit(0)

metric_label = prettify_metric_label(metric)
latexish_matplotlib_style()
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------- (1) RAW -> RAW+P ----------------
raw_results = []
for model in df["model"].unique():
    df_m = df[df["model"] == model]
    raw = df_m[df_m["scheme"]=="RAW"].set_index("seed")[metric]
    if raw.empty:
        continue
    for step in STEP_NAMES:
        rawp = df_m[df_m["scheme"]==f"RAW+{step}"].set_index("seed")[metric]
        common = raw.index.intersection(rawp.index)
        if not len(common):
            continue
        delta = rawp.loc[common] - raw.loc[common]
        raw_results.append({
            "model": model,
            "model_pretty": prettify_model(model),
            "step": step,
            "mean_delta": float(delta.mean())
        })

raw_df = pd.DataFrame(raw_results)
if not raw_df.empty:
    # orden por nombre bonito
    raw_df = raw_df.sort_values(by=["model_pretty", "step"])
    pivot_raw = raw_df.pivot(index="model_pretty", columns="step", values="mean_delta").reindex(columns=STEP_NAMES)
    pivot_raw = pivot_raw.fillna(0.0)
    pivot_raw = pivot_raw.rename(columns=STEP_LABELS)

    ax = pivot_raw.plot(kind="bar", figsize=(10, 6))
    ax.axhline(0, linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylabel(f"Δ {metric_label} (RAW+P − RAW)")
    ax.set_title("Impact of adding each step to RAW")
    ax.set_xlabel("Models")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_raw, dpi=300)  # PDF vectorial, dpi para rasterización de texto si aplica
    plt.close()
    print(f"[OK] RAW→RAW+P guardado en: {out_raw}")
else:
    print("[WARN] No se pudo calcular RAW→RAW+P (faltan pares RAW y RAW+P por seed/modelo).")

# ---------------- (2) FULL -> (FULL−P) ----------------
full_results = []
for model in df["model"].unique():
    df_m = df[df["model"] == model]
    full = df_m[df_m["scheme"]=="FULL"].set_index("seed")[metric]
    if full.empty:
        continue
    for step in STEP_NAMES:
        fullm = df_m[df_m["scheme"]==f"FULL-{step}"].set_index("seed")[metric]
        common = full.index.intersection(fullm.index)
        if not len(common):
            continue
        benefit = full.loc[common] - fullm.loc[common]  # positivo = el paso ayuda en FULL
        full_results.append({
            "model": model,
            "model_pretty": prettify_model(model),
            "step": step,
            "mean_benefit": float(benefit.mean())
        })

full_df = pd.DataFrame(full_results)
if not full_df.empty:
    full_df = full_df.sort_values(by=["model_pretty", "step"])
    pivot_full = full_df.pivot(index="model_pretty", columns="step", values="mean_benefit").reindex(columns=STEP_NAMES)
    pivot_full = pivot_full.fillna(0.0)
    pivot_full = pivot_full.rename(columns=STEP_LABELS)

    ax = pivot_full.plot(kind="bar", figsize=(10, 6))
    ax.axhline(0, linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylabel(f"Δ {metric_label} (FULL − (FULL−P))")
    ax.set_title("Impact of removing each step to FULL")
    ax.set_xlabel("Models")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_full, dpi=300)
    plt.close()
    print(f"[OK] FULL→(FULL−P) guardado en: {out_full}")
else:
    print("[WARN] No se pudo calcular FULL→(FULL−P) (faltan pares FULL y FULL−P por seed/modelo).")
