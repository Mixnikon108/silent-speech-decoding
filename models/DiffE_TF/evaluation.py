from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    *,
    num_classes: int = 5,
    k: int = 1,
    average: str = "macro",
    compute_baseline: bool = True,
    baseline_runs: int = 100,
    return_cm: bool = False,
) -> dict[str, dict[str, float] | np.ndarray]:
    """
    Evaluate a DDPM + CAE closed-vocabulary model on silent-speech data.

    The function reports the most common classification metrics for
    imagined/silent-speech EEG-BCI with *N* balanced classes and compares
    them to a uniform random baseline.

    Parameters
    ----------
    encoder : torch.nn.Module
        Feature extractor (frozen or in `eval()` mode).
    head : torch.nn.Module
        Classification head mapping encoder features → logits.
    loader : DataLoader
        Iterable dataloader providing batches ``(x, y)``.
    device : torch.device | str
        Device where `encoder` and `head` live (e.g. ``"cuda:0"``).
    num_classes : int, default=5
        Number of target classes.
    k : int, default=1
        *Top-k* for accuracy (``k=1`` → standard accuracy).
    average : {"macro", "micro", "weighted"}, default="macro"
        Averaging method for precision / recall / F1 / AUC.
    compute_baseline : bool, default=True
        If *True*, compute the mean performance of a random classifier.
    baseline_runs : int, default=100
        Number of Monte-Carlo iterations for the random baseline.
    return_cm : bool, default=False
        If *True*, include the confusion matrix in the output.

    Returns
    -------
    dict
        ``{
            "metrics":          {...},          # real model
            "baseline_random":  {...},          # averaged random baseline
            "improvement_%":    {...},          # relative gain
            ["confusion_matrix": ndarray]       # only if `return_cm`
        }``

    Notes
    -----
    * All metrics are float32 for compatibility with JSON/YAML logs.
    * Metrics identical for model & baseline share the same keys, enabling
      straightforward comparison and dashboard plotting.
    """
    # ------------------------------------------------------------------ #
    # 1. Forward pass (no gradients)                                     #
    # ------------------------------------------------------------------ #
    encoder.eval()
    head.eval()

    y_true: list[torch.Tensor] = []
    y_prob: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)

        # Forward through encoder and classification head
        logits = head(encoder(x)[0])          # type: ignore[index]
        prob = F.softmax(logits, dim=1)

        y_true.append(y.cpu())
        y_prob.append(prob.cpu())

    y_true_np = torch.cat(y_true).numpy()
    y_prob_np = torch.cat(y_prob).numpy()
    y_pred_np = y_prob_np.argmax(axis=1)

    labels = np.arange(num_classes)

    # ------------------------------------------------------------------ #
    # 2. Primary metrics                                                 #
    # ------------------------------------------------------------------ #
    metrics: dict[str, float | np.ndarray] = {
        "accuracy":           float(top_k_accuracy_score(y_true_np, y_prob_np, k=k, labels=labels)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "macro_f1":           float(f1_score(y_true_np, y_pred_np, average=average, labels=labels)),
        "macro_precision":    float(precision_score(y_true_np, y_pred_np, average=average, labels=labels, zero_division=0)),
        "macro_recall":       float(recall_score(y_true_np, y_pred_np, average=average, labels=labels)),
        "roc_auc_ovo":        float(roc_auc_score(y_true_np, y_prob_np, multi_class="ovo", average=average, labels=labels)),
        "mcc":                float(matthews_corrcoef(y_true_np, y_pred_np)),
        "cohen_kappa":        float(cohen_kappa_score(y_true_np, y_pred_np)),
    }
    if return_cm:
        metrics["confusion_matrix"] = confusion_matrix(y_true_np, y_pred_np, labels=labels)

    # ------------------------------------------------------------------ #
    # 3. Uniform-random baseline (Monte-Carlo)                            #
    # ------------------------------------------------------------------ #
    baseline: dict[str, float] = {}
    if compute_baseline:
        rng = np.random.default_rng(seed=42)
        # Prepare containers
        tmp: dict[str, list[float]] = {m: [] for m in metrics if m != "confusion_matrix"}

        for _ in range(baseline_runs):
            rand_pred = rng.integers(0, num_classes, size=y_true_np.shape)
            rand_prob = np.full((y_true_np.size, num_classes), 1.0 / num_classes)

            # Same metrics as above (except confusion matrix)
            tmp["accuracy"].append(top_k_accuracy_score(y_true_np, rand_prob, k=k, labels=labels))
            tmp["balanced_accuracy"].append(balanced_accuracy_score(y_true_np, rand_pred))
            tmp["macro_f1"].append(f1_score(y_true_np, rand_pred, average=average, labels=labels, zero_division=0))
            tmp["macro_precision"].append(precision_score(y_true_np, rand_pred, average=average, labels=labels, zero_division=0))
            tmp["macro_recall"].append(recall_score(y_true_np, rand_pred, average=average, labels=labels))
            tmp["roc_auc_ovo"].append(roc_auc_score(y_true_np, rand_prob, multi_class="ovo", average=average, labels=labels))
            tmp["mcc"].append(matthews_corrcoef(y_true_np, rand_pred))
            tmp["cohen_kappa"].append(cohen_kappa_score(y_true_np, rand_pred))

        # Mean across runs
        baseline = {k: float(np.mean(v)) for k, v in tmp.items()}

    # ------------------------------------------------------------------ #
    # 4. Relative improvement (%)                                         #
    # ------------------------------------------------------------------ #
    improvement = {
        k: (np.nan if baseline.get(k, 0.0) == 0.0
            else 100.0 * (metrics[k] - baseline[k]) / baseline[k])
        for k in baseline
    }

    # ------------------------------------------------------------------ #
    # 5. Return consolidated report                                       #
    # ------------------------------------------------------------------ #
    return {
        "metrics":          metrics,
        "baseline_random":  baseline,
        "improvement_%":    improvement,
        **({"confusion_matrix": metrics["confusion_matrix"]} if return_cm else {}),
    }
