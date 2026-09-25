"""Парсер экспертной разметки (разметка.xlsx, лист «Калибровка»).

Семь бинарных флагов на исследование × область. Пустая ячейка — «область
не снималась», а не «дефекта нет»: NaN нулём не заполняется.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "n", "study",
    "spine_coverage", "spine_axis", "spine_artifact",
    "hip_r_positioning", "hip_r_roi",
    "hip_l_positioning", "hip_l_roi",
    "verdict_spine", "verdict_hip_r", "verdict_hip_l",
    "comment",
    "_c13", "_ko_spine", "_ko_hip_r", "_ko_hip_l", "_ko_verdict", "_c18",
]

FLAGS = COLUMNS[2:9]

# Регионы и относящиеся к ним флаги — в терминах ТЗ (6 кодов), не Excel (5):
# ТЗ разводит позиционирование бедра и ротацию, разметка их склеивает.
REGION_FLAGS = {
    "spine": ["spine_coverage", "spine_axis", "spine_artifact"],
    "hip_right": ["hip_r_positioning", "hip_r_roi"],
    "hip_left": ["hip_l_positioning", "hip_l_roi"],
}

REGION_VERDICT = {
    "spine": "verdict_spine",
    "hip_right": "verdict_hip_r",
    "hip_left": "verdict_hip_l",
}

VIOLATION_CODES = {
    "spine_coverage": "spine_coverage",
    "spine_axis": "spine_axis",
    "spine_artifact": "spine_artifact",
    # Склеенный флаг: при обучении разворачивается в hip_positioning | hip_rotation
    "hip_r_positioning": "hip_positioning|hip_rotation",
    "hip_l_positioning": "hip_positioning|hip_rotation",
    "hip_r_roi": "hip_roi_coverage",
    "hip_l_roi": "hip_roi_coverage",
}


def load(xlsx: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name="Калибровка", header=None)
    df = raw.iloc[2:].copy()
    df.columns = COLUMNS[: raw.shape[1]]
    df = df[df["study"].notna()].reset_index(drop=True)
    for c in FLAGS + ["verdict_spine", "verdict_hip_r", "verdict_hip_l"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["study"] = df["study"].astype(str)
    return df[["study"] + FLAGS + ["verdict_spine", "verdict_hip_r", "verdict_hip_l", "comment"]]


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Длинная форма: строка на исследование × область, только снятые области.

    quality_class — ИЛИ флагов по критериям, а не колонка «Итог» (она в verdict_itog).
    """
    rows = []
    for _, r in df.iterrows():
        for region, flags in REGION_FLAGS.items():
            vals = {f: r[f] for f in flags}
            if all(pd.isna(v) for v in vals.values()):
                continue  # область не снималась
            violations = [VIOLATION_CODES[f] for f, v in vals.items() if v == 1]
            flags_or = int(any(v == 1 for v in vals.values()))
            verdict = r[REGION_VERDICT[region]]
            itog = int(verdict) if pd.notna(verdict) else flags_or
            rows.append({
                "study": r["study"],
                "region": region,
                "quality_class": flags_or,
                "flags_or": flags_or,
                "verdict_itog": itog,
                "verdict_mismatch": int(itog != flags_or),
                "violation_type": ";".join(violations),
                "comment": r["comment"] if isinstance(r["comment"], str) else "",
                **vals,
            })
    return pd.DataFrame(rows)
