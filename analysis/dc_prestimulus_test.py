#!/usr/bin/env python3
"""Pre-stimulus DC localization test.

Question: the raw-EEG accuracy in BCI2020 Track 3 is driven by the per-trial DC
offset. Is that offset a neural response to the imagined word, or a baseline/
acquisition effect present before the task?

This script isolates the DC (per-channel temporal mean) over three windows and
classifies the 5 words with an RBF SVM (5-fold CV):
    - full trial           (-500 -> 2600 ms)
    - pre-stimulus only    (-500 -> 0 ms, i.e. before stimulus onset)
    - post-stimulus only   (0 -> 2600 ms)
and, as a control, the DC after per-trial baseline correction.

If the pre-stimulus window already classifies above chance, the discriminative
information precedes the imagined-speech period and therefore cannot be a neural
correlate of it. On BCI2020 Track 3 (Subject 1) it does: pre-stimulus alone
reaches ~0.41 (chance 0.20), and removing the static offset collapses accuracy
to ~0.27.

Usage:
    python analysis/dc_prestimulus_test.py --mat data/raw/BCI2020/training_set/Data_Sample01.mat
"""
import argparse
import numpy as np
import scipy.io as sio
from scipy.stats import f_oneway
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

FS = 256
N_PRE = 128  # samples with t < 0 (0.5 s pre-stimulus at 256 Hz)


def to_trials_channels_samples(x):
    """Reorder an EEG array to (trials, channels, samples) using known sizes."""
    shp = x.shape
    ax_s = int(np.argmax(shp))                      # samples is the largest axis (795)
    ax_c = [i for i in range(3) if shp[i] == 64][0]  # 64 channels
    ax_t = [i for i in range(3) if i not in (ax_s, ax_c)][0]
    return np.transpose(x, (ax_t, ax_c, ax_s))


def cv_svm(features, labels, seed=42):
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mat", required=True, help="BCI2020 Track3 training .mat file for one subject")
    ap.add_argument("--key", default="epo_train", help="MATLAB struct name")
    args = ap.parse_args()

    m = sio.loadmat(args.mat, squeeze_me=False, struct_as_record=False)
    epo = m[args.key][0, 0]
    X = to_trials_channels_samples(np.asarray(epo.x, dtype=np.float64))
    y = np.argmax(np.asarray(epo.y), axis=0).ravel()
    print(f"X = {X.shape} (trials, channels, samples) | classes = {np.unique(y, return_counts=True)}")

    dc_full = X.mean(axis=2)
    dc_pre = X[:, :, :N_PRE].mean(axis=2)
    dc_post = X[:, :, N_PRE:].mean(axis=2)
    dc_bc = (X - X[:, :, :N_PRE].mean(axis=2, keepdims=True)).mean(axis=2)

    print("\nSVM-RBF, 5-fold CV (chance = 0.20)")
    for name, feats in [("DC full trial", dc_full),
                        ("DC pre-stimulus", dc_pre),
                        ("DC post-stimulus", dc_post),
                        ("DC baseline-corrected", dc_bc)]:
        mean, std = cv_svm(feats, y)
        print(f"  {name:24s}: {mean:.3f} +/- {std:.3f}")

    r = np.median([np.corrcoef(dc_pre[i], dc_post[i])[0, 1] for i in range(X.shape[0])])
    F, p = f_oneway(*[dc_full.mean(1)[y == c] for c in np.unique(y)])
    print(f"\n  median per-trial spatial corr(pre, post) = {r:.3f}  (static offset if ~1)")
    print(f"  global-DC by class ANOVA: F = {F:.2f}, p = {p:.2e}")


if __name__ == "__main__":
    main()
