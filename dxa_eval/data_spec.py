"""Какие файлы аугментации входят в обучение — явным списком, а не глобом.

Глоб делал состав выборки зависимым от случайных файлов в data/. Новый вид синтетики — впишите сюда.
"""

from __future__ import annotations

import warnings
from pathlib import Path

# Копии снимков (поворот, сдвиг, яркость) — только для голов типов бедра; годности вредят.
AUG_FILES = ("data/features_aug.csv",)

# Синтетика с изменённой меткой: поворот делает наклон оси истинным.
SYNTH_FILES = ("data/features_labelaug.csv",)

# Как пересобрать, если файлов нет:
REBUILD = {
    "data/features_aug.csv": "python scripts/pipeline/augment_features.py",
    "data/features_labelaug.csv": "python scripts/pipeline/augment_labels.py --kind tilt",
}


def resolve(names: tuple[str, ...], root: Path | None = None) -> list[Path]:
    """Существующие файлы из списка; без них метрика несравнима, поэтому громко предупреждает."""
    root = root or Path.cwd()
    out, missing = [], []
    for n in names:
        p = root / n
        (out if p.exists() else missing).append(p)
    if missing:
        how = "; ".join(REBUILD.get(str(p.relative_to(root)), "?") for p in missing)
        warnings.warn(
            f"нет файлов аугментации: {', '.join(p.name for p in missing)}. "
            f"Модель обучится без них и метрика будет НЕ сравнима с бейзлайном. "
            f"Пересобрать: {how}", RuntimeWarning, stacklevel=2)

    # Сравнение со всеми объявленными файлами, а не только запрошенными: иначе чужой список сочтётся посторонним.
    declared = {root / n for n in AUG_FILES + SYNTH_FILES}
    stray = sorted(set(root.glob("data/features_aug*.csv")) - declared)
    if stray:
        warnings.warn(
            f"в data/ лежат посторонние файлы аугментации: "
            f"{', '.join(p.name for p in stray)}. В обучение они не идут — "
            f"состав задаётся dxa_eval/data_spec.py. Удалите или впишите в список.",
            RuntimeWarning, stacklevel=2)
    return out
