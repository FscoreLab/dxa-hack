"""Инвентаризация набора: манифест, дедуп, сплит по исследованиям."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dicom import read


def scan(root: str | Path) -> pd.DataFrame:
    """Манифест всех DICOM; ключ к разметке — имя каталога, не StudyInstanceUID (перевыдан)."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*.dcm")):
        rel = p.relative_to(root)
        meta, _ = read(p)
        d = meta.as_dict()
        d["study_dir"] = rel.parts[0]
        d["rel_path"] = str(rel)
        rows.append(d)
    return pd.DataFrame(rows)


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Помечает точные пиксельные дубли. Канонический — первый по пути."""
    df = df.sort_values("rel_path").copy()
    df["dup_group"] = df.groupby("pixel_md5").ngroup()
    df["dup_count"] = df.groupby("pixel_md5")["pixel_md5"].transform("size")
    df["is_canonical"] = ~df.duplicated("pixel_md5", keep="first")
    return df


def split_by_study(
    studies: list[str], n_folds: int = 5, seed: int = 20260915
) -> pd.DataFrame:
    """Фолды по исследованиям, не по изображениям — иначе дубли протекут."""
    rng = np.random.default_rng(seed)
    order = np.array(sorted(set(studies)))
    rng.shuffle(order)
    return pd.DataFrame({"study": order, "fold": np.arange(len(order)) % n_folds})


def summarize(df: pd.DataFrame) -> str:
    uniq = df["pixel_md5"].nunique()
    per_study = df.groupby("study_dir").size()
    scale = df["scale_source"].value_counts()
    no_scale = int((df["scale_source"] == "unknown").sum())
    mm = df.loc[df["mm_per_px_x"].notna(), "mm_per_px_x"].round(3)
    lines = [
        f"файлов: {len(df)}, уникальных по пикселям: {uniq} (дублей {len(df)-uniq})",
        f"исследований: {df['study_dir'].nunique()}, изображений на исследование: "
        f"мин {per_study.min()}, медиана {int(per_study.median())}, макс {per_study.max()}",
        f"аппараты: {df['model'].value_counts().to_dict()}",
        f"версии ПО: {df['software'].value_counts().to_dict()}",
        f"локали серии: {df['series_description'].value_counts().to_dict()}",
        f"биты: {df['bits_stored'].value_counts().to_dict()}, "
        f"фотометрия: {df['photometric'].value_counts().to_dict()}",
        f"источник масштаба: {scale.to_dict()}",
        f"мм/пиксель: {mm.value_counts().head(8).to_dict()}",
        f"БЕЗ масштаба: {no_scale} файлов — сантиметровые предикаты на них не считаются",
    ]
    return "\n".join(lines)
