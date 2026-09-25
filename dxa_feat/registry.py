"""Реестр наборов признаков: каждый f_*.py в пакете подхватывается сам.

    from dxa_feat.registry import feature

    @feature(regions=("spine",), needs=("img", "mask", "discs"))
    def spine_tilt_v2(img, mask, discs):
        '''Наклон оси по крайним телам.'''
        return {"tilt_v2_deg": ...}
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Iterable

# Имя из needs совпадает с именем аргумента функции.
KNOWN_NEEDS = ("img", "mask", "img_r", "mask_r", "meta", "side", "region",
               "discs", "embedding", "sop_uid", "fold", "kp_points")

REGIONS_ALL = ("spine", "hip")


@dataclass(frozen=True)
class Extractor:
    name: str
    fn: Callable[..., dict]
    regions: tuple[str, ...]
    needs: tuple[str, ...]
    enabled: bool
    note: str = ""

    def __call__(self, ctx) -> dict:
        out = self.fn(*(getattr(ctx, n) for n in self.needs))
        # Ключи с _ — внутренние объекты, в таблицу признаков не идут.
        return {k: v for k, v in out.items() if not k.startswith("_")}


_REGISTRY: list[Extractor] = []
_LOADED = False


def feature(*, regions: Iterable[str] = REGIONS_ALL,
            needs: Iterable[str] = ("img", "mask"),
            enabled: bool = True, note: str = ""):
    """Регистрирует функцию как набор признаков; needs — аргументы из KNOWN_NEEDS по порядку."""
    regions, needs = tuple(regions), tuple(needs)
    bad = set(needs) - set(KNOWN_NEEDS)
    if bad:
        raise ValueError(f"неизвестные needs: {sorted(bad)}; есть {KNOWN_NEEDS}")
    bad = set(regions) - set(REGIONS_ALL)
    if bad:
        raise ValueError(f"неизвестные regions: {sorted(bad)}; есть {REGIONS_ALL}")

    def deco(fn):
        _REGISTRY.append(Extractor(fn.__name__, fn, regions, needs, enabled, note))
        return fn

    return deco


def _load() -> None:
    """Импортирует все dxa_feat/f_*.py — регистрация происходит при импорте."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import dxa_feat
    for mod in pkgutil.iter_modules(dxa_feat.__path__):
        if mod.name.startswith("f_"):
            importlib.import_module(f"dxa_feat.{mod.name}")


def for_region(region: str) -> list[Extractor]:
    """Включённые наборы для области; region — "spine" или что угодно ещё (бедро)."""
    _load()
    key = "spine" if region == "spine" else "hip"
    return [e for e in _REGISTRY if e.enabled and key in e.regions]


def collect(ctx, region: str) -> dict:
    """Считает все наборы области; столкновение имён колонок — ошибка, а не перезапись."""
    out: dict = {}
    owner: dict[str, str] = {}
    for ex in for_region(region):
        for k, v in ex(ctx).items():
            if k in owner:
                raise KeyError(
                    f"признак {k!r} выдают сразу два набора: {owner[k]} и "
                    f"{ex.name}. Переименуйте свой — молча затирать нельзя")
            owner[k] = ex.name
            out[k] = v
    return out


def describe() -> str:
    """Опись реестра для человека: что включено, что отвергнуто и почему."""
    _load()
    lines = []
    for e in sorted(_REGISTRY, key=lambda e: (not e.enabled, e.name)):
        mark = " " if e.enabled else "✗"
        doc = (e.fn.__doc__ or "").strip().split("\n")[0]
        lines.append(f"{mark} {e.name:<28} {'+'.join(e.regions):<10} {doc}")
        if not e.enabled and e.note:
            lines.append(f"    отвергнут: {e.note}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
