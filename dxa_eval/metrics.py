"""Оценка на малой выборке: бутстрэп по исследованиям, порог внутри обучающих фолдов.

При малом числе позитивов строка отчёта помечается как exploratory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)

MIN_POSITIVES_STABLE = 15


def pick_threshold(y: np.ndarray, p: np.ndarray, by_rate: bool = True) -> float:
    """Порог по обучающей части фолда: по доле нарушений (по умолчанию) либо по максимуму F1.

    Квантиль по доле устойчивее: максимум F1 при малом числе позитивов садится на случайную ступеньку.
    """
    if len(p) < 2 or y.sum() == 0:
        return 0.5
    if by_rate:
        return float(np.quantile(p, 1.0 - float(y.mean())))
    # Кандидаты — сами значения (или их квантили), без округления: сетка не зависит от масштаба.
    ts = np.unique(p)
    if len(ts) > 1000:
        ts = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 1000)))
    if len(ts) < 2:
        return 0.5
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(scores))])


def boot_ci(fn, y: np.ndarray, p: np.ndarray, groups: np.ndarray,
            n: int = 2000, seed: int = 20260916) -> tuple[float, float]:
    """Бутстрэп по группам (исследованиям), а не по строкам."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        gs = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in gs])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(fn(y[idx], p[idx]))
        except ValueError:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report(y: np.ndarray, p: np.ndarray, groups: np.ndarray, thr, name: str,
           model: str = "") -> dict:
    """thr — число либо массив на строку (порог, взятый вне её фолда)."""
    yhat = (p >= np.asarray(thr)).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn_) if (tp + fn_) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    lo, hi = boot_ci(roc_auc_score, y, p, groups)
    # В бутстрэп идёт бинаризованный ответ: порог-массив не совпал бы по длине с ресэмплом.
    slo, shi = boot_ci(lambda a, b: b[a == 1].mean() if (a == 1).any() else np.nan,
                       y, yhat.astype(float), groups)
    return {
        "задача": name,
        "модель": model,
        "n": len(y),
        "позитивов": int(y.sum()),
        "ROC-AUC": round(auc, 3),
        "AUC 95% ДИ": f"{lo:.2f}–{hi:.2f}",
        "PR-AUC": round(average_precision_score(y, p), 3),
        "prevalence": round(float(y.mean()), 3),
        "F1": round(f1_score(y, yhat, zero_division=0), 3),
        "чувств.": round(float(sens), 3),
        "чувств. ДИ": f"{slo:.2f}–{shi:.2f}",
        "специф.": round(float(spec), 3),
        "TP/FP/FN/TN": f"{tp}/{fp}/{fn_}/{tn}",
        "статус": "exploratory" if y.sum() < MIN_POSITIVES_STABLE else "",
    }


def as_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
