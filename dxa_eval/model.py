"""Ядро модели: отбор признаков и классификаторы.

Имя модуля зашито в models/heads.pkl: перенос файла ломает распикл весов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 20260916

# Любая колонка из разметки или производная от неё обязана быть здесь, иначе метка утечёт в признаки.
LABEL_COLS = {
    "study", "region", "quality_class", "flags_or", "verdict_mismatch",
    "verdict_itog",
    "violation_type", "comment", "_t", "_pos", "pos_t", "roi_t", "y_q", "y_p",
    "spine_coverage", "spine_axis", "spine_artifact",
    "hip_r_positioning", "hip_r_roi", "hip_l_positioning", "hip_l_roi",
    "verdict_spine", "verdict_hip_r", "verdict_hip_l",
}
# Признаки режима съёмки, а не качества: на другом аппарате шорткат исчезнет.
MODE_COLS = {"rows", "cols", "mm_per_px", "scale_known", "mm_per_px_anat", "period_px"}
# Служебные колонки замера (fold приезжает слиянием с data/folds.csv) — не признаки.
SERVICE_COLS = {"fold", "copy", "n_frames_x", "n_frames_y"}


def _drop_lmk(cols):
    """Абляция признаков по префиксу из DXA_LMK_DROP."""
    import os
    pref = os.environ.get("DXA_LMK_DROP", "")
    return [c for c in cols if not (pref and c.startswith(pref))] if pref else cols


def feature_cols(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if df[c].dtype == object:
            continue
        base = c.rsplit("_", 1)[0]
        if c in LABEL_COLS or base in LABEL_COLS or base in MODE_COLS or c in MODE_COLS:
            continue
        if c in SERVICE_COLS:
            continue
        if any(c.startswith(l + "_") for l in LABEL_COLS):
            continue
        out.append(c)
    return _drop_lmk(out)


def prune_correlated(df: pd.DataFrame, cols: list[str], thr: float = 0.97) -> list[str]:
    """Убирает почти совпадающие признаки: из пары остаётся первый по списку."""
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(-1)
    X = X.loc[:, X.std() > 1e-9]
    cols = [c for c in cols if c in X.columns]
    c = np.nan_to_num(X.corr().abs().values, nan=0.0)
    idx = {n: i for i, n in enumerate(cols)}
    keep: list[str] = []
    for n in cols:
        if all(c[idx[n], idx[k]] <= thr for k in keep):
            keep.append(n)
    return keep


class MeanOfTwo:
    """Среднее бустинга и логрегрессии: выбор одной из двух на малой выборке — шум."""

    def __init__(self):
        self.a = make_model("hgb")
        self.b = make_model("logreg")

    def fit(self, X, y):
        self.a.fit(X, y)
        self.b.fit(X, y)
        return self

    def predict_proba(self, X):
        p = 0.5 * (self.a.predict_proba(X)[:, 1] + self.b.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def make_model(kind: str):
    if kind == "mean2":
        return MeanOfTwo()
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=2, max_iter=150, learning_rate=0.06,
            min_samples_leaf=8, l2_regularization=1.0, random_state=RANDOM_STATE)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=0.3, class_weight="balanced"))


