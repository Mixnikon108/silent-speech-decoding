#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figure · 3x2 layout (EEG RAW, BCI2020)
Top row: time-domain signals (3 subjects, 3 distributed channels per plot)
Bottom row: PSD via Welch (y-log), frequency axis linear 0–128 Hz for the same channels/subjects
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch  # <-- PSD (µV²/Hz)

# ======================= Configuration =======================
# Ruta al dataset relativa a la raíz del proyecto (data/processed/BCI2020/...)
RAW_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "BCI2020" / "BCI_raw.npz"

# Three subjects (columns across both rows)
SUBJECT_IDS = [1, 2, 3]

# Three distributed channels across 64-ch montage
CH_IDX    = [2, 32, 62]            # e.g., roughly frontal/central/occipital spread
CH_LABELS = ["F7", "AF7", "PO4"]   # renamed

FS = 256                 # Hz
NYQUIST = FS / 2         # 128 Hz
N_TRIALS_AVG = 10        # trials to average in PSD for representativeness
N_PERSEG = FS * 2        # 2-second Welch windows (adjust if you want)

OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "BCI2020_raw_3x2_time_psd_ylog_0to128.pdf"

# ======================= Style (LaTeX/serif) =======================
def setup_matplotlib_style():
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
    except Exception:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
    # Finer grid lines globally
    plt.rcParams["grid.linewidth"] = 0.25

# ======================= Data loading =======================
from utils import load_data_BCI  # must return: (X_train, y_train, X_val, y_val, X_test, y_test)

def load_subject_train(subject_id: int):
    Xtr, ytr, Xva, yva, Xte, yte = load_data_BCI(RAW_FILE, subject_id=subject_id)
    # Expected: Xtr shape = (trials, channels, samples)
    return Xtr

# ======================= Signal utilities =======================
def time_axis(n_samples: int, fs: int):
    return np.arange(n_samples) / fs

def avg_psd_welch(x_tc, fs, ch_idx, n_trials_avg=10, nperseg=None):
    """
    Compute average PSD (Welch) across trials.
    Units: µV²/Hz if input x is in µV.
    x_tc: (T, C, S)
    Returns freqs (0..Nyquist) and PSD array (len(ch_idx) x F)
    """
    T = min(n_trials_avg, x_tc.shape[0]) if n_trials_avg else x_tc.shape[0]
    x = x_tc[:T, :, :]  # (T, C, S)
    psd_stack = []
    for t in range(T):
        psd_channels = []
        for ch in ch_idx:
            f, pxx = welch(x[t, ch, :], fs=fs, nperseg=nperseg)
            psd_channels.append(pxx)
        psd_stack.append(np.vstack(psd_channels))  # (len(ch_idx), F)
    psd_avg = np.mean(psd_stack, axis=0)          # average over trials → (len(ch_idx), F)
    return f, np.maximum(psd_avg, 1e-16)          # clip tiny values for log-scale stability

# ======================= Figure (3x2) =======================
def plot_3x2_time_and_psd(subject_ids, ch_idx, ch_labels, fs, out_pdf):
    setup_matplotlib_style()

    # Consistent colors for the 3 channels across all subplots
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    if color_cycle is None or len(color_cycle) < len(ch_idx):
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.2, 4.6), constrained_layout=True)

    # ---------- Top row: time-domain (Amplitude in µV) ----------
    for col, sid in enumerate(subject_ids):
        Xtr = load_subject_train(sid)
        x = Xtr[0, :, :]                # trial 0 for display
        t = time_axis(x.shape[-1], fs)

        ax = axs[0, col]
        for k, ch in enumerate(ch_idx):
            ax.plot(t, x[ch, :], linewidth=0.5, label=ch_labels[k], color=color_cycle[k])
        ax.set_title(f"Subject {sid} · Time")
        if col == 0:
            ax.set_ylabel("Amplitude (µV)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3, linewidth=0.25)
        if col == 0:
            ax.legend(
                frameon=False,
                ncol=3,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                borderaxespad=0.0,
                handlelength=2,
                handletextpad=0.4,
                columnspacing=0.8
            )

    # ---------- Bottom row: PSD (y-log, x linear 0–128 Hz) ----------
    for col, sid in enumerate(subject_ids):
        Xtr = load_subject_train(sid)
        freqs, psd = avg_psd_welch(Xtr, fs=FS, ch_idx=ch_idx,
                                   n_trials_avg=N_TRIALS_AVG, nperseg=N_PERSEG)

        ax = axs[1, col]
        for k in range(len(ch_idx)):
            ax.plot(freqs, psd[k, :], linewidth=0.5, label=ch_labels[k], color=color_cycle[k])
        ax.set_title(f"Subject {sid} · PSD (Welch)")
        if col == 0:
            ax.set_ylabel("PSD (µV²/Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, FS/2)            # 0 … 128 Hz
        ax.set_yscale("log")            # PSD in log scale
        ax.grid(True, which="both", alpha=0.3, linewidth=0.25)
        if col == 0:
            ax.legend(
                frameon=False,
                ncol=3,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                borderaxespad=0.0,
                handlelength=2,
                handletextpad=0.4,
                columnspacing=0.8
            )

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

# ======================= Main =======================
if __name__ == "__main__":
    assert len(CH_IDX) == 3, "Please provide exactly 3 channels."
    plot_3x2_time_and_psd(
        subject_ids=SUBJECT_IDS,
        ch_idx=CH_IDX,
        ch_labels=CH_LABELS,
        fs=FS,
        out_pdf=OUT_PDF
    )
