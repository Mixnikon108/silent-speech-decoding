import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# Estadística / ML
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.stats.multitest import multipletests

# Utilidades
import json, os

# ==== Configuración ====
DATA_ROOT = Path("/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020")
RAW_FILE = DATA_ROOT / "BCI_raw.npz"       # Cambia si procede
PROC_FILE = DATA_ROOT / "BCI_LP005.npz"    # Opcional, no se usa para tests analíticos
SUBJECT_ID = 1
FS = 256
LOW_BANDS = [(0.0,1.0),(0.5,4.0),(4.0,8.0)]  # ULF/delta/theta de ejemplo
N_MI_BINS = 16
N_PERMUTATIONS = 200  # para tests de permutación (ajusta si quieres más potencia)
CV_FOLDS = 5
RANDOM_STATE = 42
OUTDIR = Path("./analysis_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ==== Carga ====
from utils import load_data_BCI  # Debes tener esta función
X_train, y_train, X_val, y_val, X_test, y_test = load_data_BCI(RAW_FILE, subject_id=SUBJECT_ID)

# Asumimos shapes: X_*: (n_trials, n_channels, T); y_*: (n_trials,)
def to_numpy_int(y):
    y = np.asarray(y)
    if y.ndim>1:
        y = y.squeeze()
    # garantizar entero [0..K-1]
    classes, y_enc = np.unique(y, return_inverse=True)
    return y_enc, classes

y_train_enc, classes = to_numpy_int(y_train)
y_val_enc, _ = to_numpy_int(y_val)
y_test_enc, _ = to_numpy_int(y_test)
n_classes = len(classes)

n_tr, n_ch, T = X_train.shape
print(f"Train shape: {X_train.shape}; classes={n_classes}, chans={n_ch}, T={T}")

# ==== Helpers ====
def bandpower_fft(x, fs, f_lo, f_hi):
    # x: (..., T)
    Xf = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(x.shape[-1], d=1/fs)
    m = (freqs >= f_lo) & (freqs < f_hi)
    # potencia = suma |X|^2 / N (sin normalizar por ventana; consistente para comparar)
    p = (np.abs(Xf[..., m])**2).sum(axis=-1)
    return p

def slope_and_offset(x):
    # Ajuste lineal por mínimos cuadrados por canal y trial: x_t = a*t + b
    # Devuelve (a,b)
    t = np.arange(x.shape[-1])
    t = (t - t.mean()) / t.std()  # normalizar t para estabilidad
    # a = cov(x,t)/var(t), b = mean(x) (porque t centrado)
    # Broadcast: x: (..., T), t: (T,)
    xm = x.mean(axis=-1, keepdims=True)
    cov = ((x - xm) * t).sum(axis=-1)
    var_t = (t**2).sum()
    a = cov / var_t
    b = xm.squeeze(-1)
    return a, b

def fdr(pvals, alpha=0.05, method="fdr_bh"):
    rej, p_corr, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return rej, p_corr

def eta_squared_from_kw(H, k, N):
    # Aproximación de tamaño de efecto para Kruskal–Wallis
    # Referencia común: eta^2_KW = (H - k + 1) / (N - k)
    return max(0.0, (H - k + 1.0) / (N - k)) if N>k else np.nan

def vectorize_trials(X):
    # Aplana (n_trials, n_channels, T) -> (n_trials, n_channels*T)
    n, c, t = X.shape
    return X.reshape(n, c*t)

def cv_score_model(X, y, clf):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=None)
    return scores.mean(), scores.std()

rng = np.random.default_rng(RANDOM_STATE)

# ==== (1) Importancia de electrodos: tests por canal ====
# Features por canal y trial: offset, slope, bandpowers bajas
a, b = slope_and_offset(X_train)  # a: (n_trials, n_channels), b igual
features = defaultdict(list)
pvals_kw = []
etas = []
channels = list(range(n_ch))

# construir matriz features_ch: (n_trials, n_features) para cada canal, y hacer KW contra y
feat_names = [f"BP_{lo:.1f}-{hi:.1f}Hz" for (lo,hi) in LOW_BANDS] + ["slope","offset"]
per_channel_results = []

for ch in channels:
    feats_ch = []
    # bandpowers
    for (lo,hi) in LOW_BANDS:
        bp = bandpower_fft(X_train[:, ch, :], FS, lo, hi)
        feats_ch.append(bp)
    # slope & offset
    feats_ch.append(a[:, ch])
    feats_ch.append(b[:, ch])
    feats_ch = np.vstack(feats_ch).T  # (n_trials, n_features)
    # Kruskal–Wallis univar por feature y Stouffer combine p-values o min-p?
    # Usamos min-p: más conservador para señal.
    Hs, ps = [], []
    for fi in range(feats_ch.shape[1]):
        groups = [feats_ch[y_train_enc==cls, fi] for cls in range(n_classes)]
        H, p = kruskal(*groups)
        Hs.append(H); ps.append(p)
    p_min = np.min(ps)
    pvals_kw.append(p_min)
    # tamaño de efecto aproximado usando H más grande
    H_max = float(np.max(Hs))
    eta = eta_squared_from_kw(H_max, n_classes, len(y_train_enc))
    etas.append(eta)

    # MI multivariada (discretizando con quantiles internos de sklearn)
    # Para estabilidad, estandarizamos feats_ch
    Xf = StandardScaler().fit_transform(feats_ch)
    mi = mutual_info_classif(Xf, y_train_enc, discrete_features=False, random_state=RANDOM_STATE)
    mi_sum = float(np.sum(mi))

    per_channel_results.append({
        "channel": ch,
        "kw_min_p": float(p_min),
        "kw_eta2_from_maxH": eta,
        "mi_sum": mi_sum,
        **{f"MI_{fn}": float(miv) for fn, miv in zip(feat_names, mi)}
    })

rej_kw, p_kw_corr = fdr(pvals_kw, alpha=0.05)
for i, d in enumerate(per_channel_results):
    d["kw_p_adj"] = float(p_kw_corr[i])
    d["kw_significant"] = bool(rej_kw[i])

# Guardar CSV
import pandas as pd
df_ch = pd.DataFrame(per_channel_results).sort_values(
    by=["kw_significant","kw_p_adj","mi_sum","kw_eta2_from_maxH"],
    ascending=[False, True, False, False]
)
df_ch.to_csv(OUTDIR / "channel_importance_stats.csv", index=False)

print("\nTop canales por evidencia (Kruskal FDR + MI):")
print(df_ch.head(10)[["channel","kw_p_adj","kw_significant","kw_eta2_from_maxH","mi_sum"]])

# ==== (1b) Ablaciones de identidad de canal con un clasificador sencillo ====
def evaluate_with_channel_permutation(X, y, mode="none"):
    """
    mode:
      - "none": sin permutar
      - "fixed": una permutación fija de canales igual para todos los trials
      - "per_trial": permutación aleatoria independiente por trial (destruye identidad)
    """
    Xp = X.copy()
    if mode == "fixed":
        perm = rng.permutation(X.shape[1])
        Xp = Xp[:, perm, :]
    elif mode == "per_trial":
        for i in range(X.shape[0]):
            perm = rng.permutation(X.shape[1])
            Xp[i] = Xp[i, perm, :]

    Xv = vectorize_trials(Xp)
    Xv = StandardScaler(with_mean=True, with_std=True).fit_transform(Xv)
    clf = LogisticRegression(
        penalty="l2", solver="saga", C=1.0, max_iter=2000, random_state=RANDOM_STATE, n_jobs=None
    )
    mu, sd = cv_score_model(Xv, y, clf)
    return mu, sd

abl_results = []
for mode in ["none","fixed","per_trial"]:
    mu, sd = evaluate_with_channel_permutation(X_train, y_train_enc, mode=mode)
    abl_results.append({"mode": mode, "cv_acc_mean": mu, "cv_acc_std": sd})
df_abl = pd.DataFrame(abl_results)
df_abl.to_csv(OUTDIR / "channel_identity_ablation.csv", index=False)
print("\nAblaciones de identidad de canal (CV accuracy):")
print(df_abl)

# Interpretación:
# - none vs fixed ~ igual → el índice del canal no importa por sí mismo (el modelo es indiferente a reordenar consistentemente).
# - none >> per_trial → dependencia fuerte de identidad del canal a nivel de trial.

# ==== (2) Señal temporal / posibles bloques ====
# Asumimos que X_train está en orden “temporal” original (como vienen los trials).
# Si no lo está, sustituye por la secuencia original.

y_seq = y_train_enc.copy()
n = len(y_seq)

# (2a) Autocorrelación por lags y test por permutación
def label_autocorr(y, max_lag=10):
    # codificamos clases a enteros 0..K-1 y centramos
    y_num = y.astype(float)
    y_num = (y_num - y_num.mean()) / (y_num.std() + 1e-9)
    ac = []
    for lag in range(1, max_lag+1):
        v = y_num[:-lag] * y_num[lag:]
        ac.append(float(v.mean()))
    return np.array(ac)  # shape (max_lag,)

def perm_pvalue_stat(y, stat_fn, n_perm=N_PERMUTATIONS, rng=None):
    rng = rng or np.random.default_rng()
    observed = stat_fn(y)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        val = stat_fn(yp)
        if abs(val) >= abs(observed):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return observed, p

ac = label_autocorr(y_seq, max_lag=10)
# p-valor para lag-1 usando permutación
obs_lag1, p_lag1 = perm_pvalue_stat(y_seq, lambda y: label_autocorr(y, max_lag=1)[0], n_perm=N_PERMUTATIONS, rng=rng)

# (2b) Runs test (Wald–Wolfowitz) para 2+ clases: reducimos a binario por “clase mayoritaria vs resto”
maj = Counter(y_seq).most_common(1)[0][0]
y_bin = (y_seq == maj).astype(int)

def runs_test_binary(x):
    # x: 0/1
    runs = 1 + np.sum(x[1:] != x[:-1])
    n1 = x.sum()
    n2 = len(x) - n1
    if n1 == 0 or n2 == 0:
        return {"runs": runs, "z": np.nan, "p": 1.0}
    mu = 1 + 2*n1*n2/(n1+n2)
    var = (2*n1*n2*(2*n1*n2 - n1 - n2)) / (((n1+n2)**2) * (n1+n2-1))
    z = (runs - mu) / np.sqrt(var + 1e-9)
    # p-valor bilateral aproximando normal
    from scipy.stats import norm
    p = 2*(1 - norm.cdf(abs(z)))
    return {"runs": int(runs), "z": float(z), "p": float(p), "n1": int(n1), "n2": int(n2)}

runs_res = runs_test_binary(y_bin)

# (2c) MI entre índice temporal y etiqueta
t_idx = np.arange(n).reshape(-1,1)
mi_time = mutual_info_classif(t_idx, y_seq, discrete_features=True, random_state=RANDOM_STATE)
# Permutación para p-valor
def mi_stat(y):
    return float(mutual_info_classif(t_idx, y, discrete_features=True, random_state=RANDOM_STATE)[0])
obs_mi, p_mi = perm_pvalue_stat(y_seq, mi_stat, n_perm=N_PERMUTATIONS, rng=rng)

# (2d) Clasificador con solo “tiempo”
def eval_time_only_classifier(y, n_bins=10):
    # features: [t_norm, bin_index (one-hot)]
    t_norm = (t_idx - t_idx.mean()) / (t_idx.std() + 1e-9)
    bins = np.digitize(t_idx.flatten(), np.linspace(0, n-1, n_bins+1)[1:-1])
    X_feat = np.hstack([t_norm, bins.reshape(-1,1)])
    scaler = StandardScaler().fit(X_feat)
    Xs = scaler.transform(X_feat)
    clf = LogisticRegression(
        penalty="l2", solver="lbfgs", multi_class="auto",
        max_iter=2000, random_state=RANDOM_STATE
    )
    mu, sd = cv_score_model(Xs, y, clf)
    return mu, sd

time_only_mu, time_only_sd = eval_time_only_classifier(y_seq, n_bins=min(10, max(3, n//20)))

# (2e) CUSUM sobre proporción de clase mayoritaria + permutación
def cusum_binary(x):
    # x: 0/1, desviación de la proporción global
    p_hat = x.mean()
    s = np.cumsum(x - p_hat)
    # estadístico: rango máx-min
    return float(s.max() - s.min())

obs_cusum, p_cusum = perm_pvalue_stat(y_bin, cusum_binary, n_perm=N_PERMUTATIONS, rng=rng)

# ==== Guardar resultados temporales ====
summary_temporal = {
    "autocorr_lag1": float(obs_lag1),
    "autocorr_lag1_p": float(p_lag1),
    "autocorr_lags_1to10": [float(v) for v in ac],
    "runs_test": runs_res,
    "MI_time": float(obs_mi),
    "MI_time_p": float(p_mi),
    "time_only_classifier_cv_acc_mean": float(time_only_mu),
    "time_only_classifier_cv_acc_std": float(time_only_sd),
    "cusum_binary_stat": float(obs_cusum),
    "cusum_binary_p": float(p_cusum),
    "n_trials": int(n),
    "n_classes": int(n_classes),
    "majority_class": int(maj),
}

with open(OUTDIR / "temporal_structure_summary.json", "w") as f:
    json.dump(summary_temporal, f, indent=2)

print("\n=== RESULTADOS TEMPORALES (resumen) ===")
for k,v in summary_temporal.items():
    if isinstance(v, (float,int,str)):
        print(f"{k}: {v}")
print("Detalle completo en temporal_structure_summary.json")

# ==== Interpretación rápida impresa ====
def chance_level(c):
    return 1.0/c if c>0 else np.nan

print("\n=== INTERPRETACIÓN RÁPIDA ===")
print("- Si muchos canales tienen kw_p_adj < 0.05 y MI alta → hay señal discriminativa asociada a canales concretos.")
print("- En ablation: si acc(none) ≈ acc(fixed) pero acc(per_trial) cae → el modelo depende de la identidad de canal.")
print(f"- Time-only clf acc ≈ {time_only_mu:.3f} (azar ≈ {chance_level(n_classes):.3f}). Si >> azar → hay patrón temporal explotable.")
print(f"- Autocorr lag1 = {obs_lag1:.4f} (p={p_lag1:.3f}); MI(time)={obs_mi:.4f} (p={p_mi:.3f}); CUSUM p={p_cusum:.3f}.")
print("- Runs test p<0.05 sugiere no-aleatoriedad (posibles bloques).")
print(f"\nCSV guardados en: {OUTDIR.resolve()}")