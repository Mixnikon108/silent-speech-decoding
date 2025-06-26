"""Utility functions for loading and preprocessing the TOL Inner-Speech dataset."""

from __future__ import annotations

import pathlib
import os
from typing import List, Tuple

import mne
import numpy as np


# -----------------------------------------------------------------------------#
#                               Helper functions                               #
# -----------------------------------------------------------------------------#
def subject_name(idx: int) -> str:
    """Return standardized subject name, e.g. sub-01, sub-12."""
    return f"sub-{idx:02d}"

def select_time_window(
    data: np.ndarray,
    *,
    fs: int = 256,
    t_start: float = 1.0,
    t_end: float = 2.5,
) -> np.ndarray:
    """Crop trials to the requested time window [t_start, t_end] (seconds)."""
    start, end = int(t_start * fs), int(t_end * fs)
    return data[:, :, start:end]


def filter_by_condition(
    data: np.ndarray,
    events: np.ndarray,
    condition: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only trials of a given condition (Pron / Inner / Vis)."""

    condition = condition.upper()
    if condition == "ALL":
        return data, events

    idx = {"PRON": 0, "INNER": 1, "VIS": 2}.get(condition)
    if idx is None:
        raise ValueError(f"Unknown condition: {condition}")

    mask = events[:, 2] == idx
    return data[mask], events[mask]


def filter_by_class(
    data: np.ndarray,
    events: np.ndarray,
    klass: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only trials of a given class (Up / Down / Right / Left)."""
    klass = klass.upper()
    if klass == "ALL":
        return data, events

    idx = {"UP": 0, "DOWN": 1, "RIGHT": 2, "LEFT": 3}.get(klass)
    if idx is None:
        raise ValueError(f"Unknown class: {klass}")

    mask = events[:, 1] == idx
    return data[mask], events[mask]


def transform_for_classifier(
    data: np.ndarray,
    events: np.ndarray,
    classes: List[List[str]],
    conditions: List[List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y for a classifier.

    Each `conditions[i][j]` is paired with `classes[i][j]` and mapped to label i.
    """
    if len(classes) != len(conditions):
        raise ValueError("`classes` and `conditions` must have same outer length")

    x_list, y_list = [], []

    for group_idx, (cls_group, cnd_group) in enumerate(zip(classes, conditions)):
        if len(cls_group) != len(cnd_group):
            raise ValueError("Condition / class pairs must be aligned")

        for cls, cnd in zip(cls_group, cnd_group):
            x_tmp, y_tmp = filter_by_condition(data, events, cnd)
            x_tmp, _ = filter_by_class(x_tmp, y_tmp, cls)
            x_list.append(x_tmp)
            y_list.append(np.full(x_tmp.shape[0], group_idx))

    return np.vstack(x_list), np.concatenate(y_list)


# -----------------------------------------------------------------------------#
#                       Low-level loading helpers (MNE)                        #
# -----------------------------------------------------------------------------#
def _events_file(root: pathlib.Path, subj: str, ses: int) -> pathlib.Path:
    return root / "derivatives" / subj / f"ses-0{ses}" / f"{subj}_ses-0{ses}_events.dat"


def _epochs_file(root: pathlib.Path, subj: str, ses: int, kind: str) -> pathlib.Path:
    return (
        root
        / "derivatives"
        / subj
        / f"ses-0{ses}"
        / f"{subj}_ses-0{ses}_{kind}-epo.fif"
    )


def load_events(root: pathlib.Path, subj_idx: int, ses: int) -> np.ndarray:
    """Load *.dat events for one session."""
    return np.load(_events_file(root, subject_name(subj_idx), ses), allow_pickle=True)


def extract_subject(
    root: pathlib.Path,
    subj_idx: int,
    *,
    kind: str = "eeg",
    sessions: Tuple[int, int, int] = (1, 2, 3),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all requested sessions for a subject and stack trials."""
    kind = kind.lower()
    if kind not in {"eeg", "exg", "baseline"}:
        raise ValueError(f"Invalid datatype: {kind}")

    subj = subject_name(subj_idx)
    data, events = [], []

    for ses in sessions:
        events.append(load_events(root, subj_idx, ses))
        epochs = mne.read_epochs(
            _epochs_file(root, subj, ses, kind), verbose="WARNING"
        ).get_data()
        data.append(epochs)

    return np.vstack(data), np.vstack(events)


# -----------------------------------------------------------------------------#
#                                     main                                     #
# -----------------------------------------------------------------------------#

DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "TOL"

FS = 256
T_START, T_END = 1.5, 3.5
SUBJECTS = range(1, 11)

all_x, all_y, subj_id = [], [], []

print("Cargando dataset TOL...")

for subj in SUBJECTS:
    print(f"Cargando sujeto {subj:02d}…")
    try:
        x, y = extract_subject(DATA_ROOT, subj, kind="eeg")
        x = select_time_window(x, fs=FS, t_start=T_START, t_end=T_END)

        classes = [["Up"], ["Down"], ["Right"], ["Left"]]
        conditions = [["Inner"]] * 4
        x, y_lbl = transform_for_classifier(x, y, classes, conditions)

        all_x.append(x)
        all_y.append(y_lbl)
        subj_id.append(np.full_like(y_lbl, subj))
        print(f"   ↳ {x.shape[0]} trials")
    except Exception as exc:
        print(f"   ⚠️  skipped (reason: {exc})")

if not all_x:
    raise RuntimeError("No subjects processed successfully.")

X_all = np.vstack(all_x)
Y_all = np.concatenate(all_y)
Subjects_all = np.concatenate(subj_id)

print("\n Dimensiones del dataset TOL:")
print(f"  - Total trials : {X_all.shape[0]}")
print(f"  - Data shape   : {X_all.shape}  # trials × channels × samples")
print(f"  - Labels shape : {Y_all.shape}")
print(f"  - Subjects     : {Subjects_all.shape}")


output_path = (
pathlib.Path(__file__).resolve().parent.parent / "data" / "processed" / "TOL" / "inner_speech_all_subjects.npz"
)

np.savez(output_path, X=X_all, Y=Y_all, Subjects=Subjects_all)
print(f"\n✅ Dataset TOL guardado en: {output_path}")
