"""Вердикт по области: пороговые правила по физическим величинам и обученные модели («головы»).

«Головы» — независимые модели на общем наборе признаков, а не выходы общего бэкбона.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dxa_service.quality import (SPINE_QUALITY_W, hip_score, load_refs,
                                 spine_score, spine_tilt)

# Критерий методических рекомендаций НПКЦ ДиТ, вып. 110.
METHODOLOGY_TILT_DEG = 5.0


def _tilt_threshold() -> float:
    """Порог наклона, откалиброванный под наш измеритель: он систематически ниже врачебного."""
    p = Path(__file__).resolve().parents[1] / "models" / "tilt_threshold.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["max_spine_tilt_deg"])
    return METHODOLOGY_TILT_DEG


MAX_SPINE_TILT_DEG = _tilt_threshold()


def _vb_limit() -> float:
    """Порог для оси по центрам тел; калибруется составная величина, которую применяет правило."""
    import os

    if "DXA_VB_LIMIT" in os.environ:
        return float(os.environ["DXA_VB_LIMIT"])
    p = Path(__file__).resolve().parents[1] / "models" / "tilt_threshold.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if "vb_axis_limit_deg" in d:
            return float(d["vb_axis_limit_deg"])
    return MAX_SPINE_TILT_DEG


VB_AXIS_LIMIT_DEG = _vb_limit()


def _artifact_rule() -> tuple[str, float] | None:
    """Порог на посторонние структуры (доля ярких дуг), подобранный по обучающей выборке."""
    p = Path(__file__).resolve().parents[1] / "models" / "artifact_threshold.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["feature"], float(d["threshold"])


ARTIFACT_RULE = _artifact_rule()


def default_rules() -> dict:
    """Пороги правил словарём: в замере их подменяют значениями, подобранными внутри фолда."""
    return {"vb_axis_limit": VB_AXIS_LIMIT_DEG,
            "max_spine_tilt": MAX_SPINE_TILT_DEG,
            "artifact_feature": ARTIFACT_RULE[0] if ARTIFACT_RULE else None,
            "artifact_threshold": ARTIFACT_RULE[1] if ARTIFACT_RULE else None}

# Тип нарушения → имя головы. Ось и артефакты — на правилах.
SPINE_HEADS = (("spine_coverage", "spine_coverage"),)

HIP_HEADS = (("hip_positioning", "hip_positioning"), ("hip_roi_coverage", "hip_roi"))


class RuleModel:
    def __init__(self, heads: dict | None = None, refs: dict | None = None,
                 rules: dict | None = None):
        self.heads = heads or {}
        # опорные распределения для рангов; None — сервис уйдёт на запасную шкалу
        self.refs = refs if refs is not None else load_refs()
        self.rules = {**default_rules(), **(rules or {})}

    @classmethod
    def load(cls, path: str | Path = "models/heads.pkl"):
        """Загружает модели; другая версия sklearn молча меняет предсказания, поэтому предупреждаем."""
        import pickle
        import warnings

        import sklearn

        p = Path(path)
        if not p.exists():
            return cls({})
        heads = pickle.load(open(p, "rb"))
        env = heads.pop("__env__", None)
        if env and env.get("sklearn") != sklearn.__version__:
            warnings.warn(
                f"модели обучены на sklearn {env['sklearn']}, здесь "
                f"{sklearn.__version__}: предсказания будут отличаться от тех, "
                "на которых измерено качество", RuntimeWarning, stacklevel=2)
        return cls(heads)

    @staticmethod
    def _num(v) -> float:
        """Нечисловое или бесконечное значение → -1, как в обучении; NaN роняет модели."""
        try:
            x = float(v)
        except (TypeError, ValueError):
            return -1.0
        return x if np.isfinite(x) else -1.0

    @classmethod
    def _value(cls, f: dict, col: str) -> float:
        """Значение колонки с учётом суффиксов агрегата (_mean/_min/_max) в обе стороны."""
        if col in f:
            return cls._num(f[col])
        base, _, suf = col.rpartition("_")
        if suf in ("mean", "min", "max") and base in f:
            return cls._num(f[base])
        # Обратное направление: правило просит базовое имя, а в агрегате лежит с суффиксом.
        for s in ("mean", "max", "min"):
            if f"{col}_{s}" in f:
                return cls._num(f[f"{col}_{s}"])
        return -1.0

    def _head(self, name: str, f: dict) -> bool:
        h = self.heads.get(name)
        if not h:
            return False
        return self._proba(name, f) >= h["threshold"]

    def _proba(self, name: str, f: dict) -> float:
        h = self.heads.get(name)
        if not h:
            return 0.0
        x = np.array([[self._value(f, c) for c in h["features"]]])
        return float(h["model"].predict_proba(x)[0, 1])

    def _quality_components(self, f: dict) -> tuple[float, float, float, float]:
        """Наклон, дуги, охват, голова — ровно те величины, что и в замере."""
        tilt = spine_tilt(self._value(f, "vb_axis_deg"),
                          self._value(f, "col_tilt_deg"),
                          self._value(f, "axis_deg_max"))
        arc = self._value(f, self.rules.get("artifact_feature") or "kp_art_frac_mean")
        # Пропуски как в замере: дуги 0.0, охват 0.5.
        arc = 0.0 if arc < 0 else arc
        cov = self._proba("spine_coverage", f) if "spine_coverage" in self.heads else 0.5
        head = self._proba("spine_quality", f) if "spine_quality" in self.heads else 0.5
        return tilt, arc, cov, head

    def quality_proba(self, region: str, f: dict, violations: list[str]) -> float:
        """Непрерывная оценка годности для ROC-AUC; считается тем же кодом, что и в замере."""
        # К вердикту не подгоняется: у пограничных случаев знак может не совпасть.
        refs = self.refs
        if region == "spine":
            tilt, arc, cov, head = self._quality_components(f)
            if refs is None:
                return self._legacy_spine_proba(tilt, arc, cov, head)
            return float(spine_score(tilt, arc, cov, head, refs))
        head = self._proba("hip_quality", f)
        return head if refs is None else float(hip_score(head, refs))

    def _legacy_spine_proba(self, tilt: float, arc: float,
                            cov: float, head: float) -> float:
        """Линейная запасная шкала без models/quality_refs.json — не та, на которой измерено качество."""
        import warnings
        warnings.warn("models/quality_refs.json не найден: непрерывная оценка "
                      "годности считается запасной шкалой, а не той, на которой "
                      "измерено качество", RuntimeWarning, stacklevel=2)
        p = [cov]
        thr = self.rules["artifact_threshold"]
        if thr:
            p.append(min(max(arc / (2 * thr), 0.0), 1.0))
        p.append(min(max(tilt / (2 * self.rules["max_spine_tilt"]), 0.0), 1.0))
        base = max(p) if p else 0.0
        if "spine_quality" in self.heads:
            base = (1.0 - SPINE_QUALITY_W) * base + SPINE_QUALITY_W * head
        return base

    def quality(self, region: str, f: dict, violations: list[str]) -> int:
        """Годность: позвоночник — «есть хотя бы один тип», бедро — своя голова (так точнее)."""
        if region == "spine":
            return int(bool(violations))
        return int(self._head("hip_quality", f))

    def reconcile_hip(self, f: dict, violations: list[str], q: int) -> list[str]:
        """Типы бедра согласуются с годностью, которую выносит своя голова.

        Годность бедра и типы — разные головы, и без согласования строка отчёта
        противоречит сама себе: «нарушение» без типа (6 областей из 150) или
        тип при «годно» (7). Годность главнее — она точнее объединения типов.
        Нарушение без типа получает тип, чья голова ближе к своему порогу;
        при «годно» типы снимаются. Замер: F1 годности не меняется, macro-F1
        по типам +0,002 (укладка −0,026, область интереса +0,038).
        """
        if not q:
            return []
        if violations:
            return violations
        best = max(HIP_HEADS, key=lambda vh: self._proba(vh[1], f)
                   / max(float(self.heads.get(vh[1], {}).get("threshold", 0.5)), 1e-6)
                   if vh[1] in self.heads else -1.0)
        return [best[0]]

    def predict(self, region: str, f: dict) -> list[str]:
        out = []
        if region == "spine":
            # Признаки читать через _value: приходит агрегат с суффиксами.
            # Угол по центрам тел идёт первым, как в методике: яркостный центр при сколиозе тянут отростки.
            import os
            from dxa_service.quality import spine_tilt
            flat = self.rules["max_spine_tilt"]
            # Та же величина, что калибрует порог и идёт в непрерывную оценку.
            vb = (self._value(f, "vb_axis_deg")
                  if os.environ.get("DXA_VB_AXIS", "1") == "1" else -1.0)
            tilt = spine_tilt(vb, self._value(f, "col_tilt_deg"),
                              self._value(f, "axis_deg_max"))
            limit = self.rules["vb_axis_limit"] if vb >= 0 else flat
            if tilt > limit:
                out.append("spine_axis")
            name, thr = self.rules["artifact_feature"], self.rules["artifact_threshold"]
            if name:
                # Имя признака хранится с суффиксом агрегата.
                if self._value(f, name.rsplit("_", 1)[0]) > thr:
                    out.append("spine_artifact")
            elif self._head("spine_artifact", f):
                out.append("spine_artifact")
            for violation, head in SPINE_HEADS:
                if self._head(head, f):
                    out.append(violation)
        else:
            for violation, head in HIP_HEADS:
                if self._head(head, f):
                    out.append(violation)
        return out

    def save(self, path: str | Path) -> None:
        """Фиксирует пороги решения — нужно для воспроизводимости по ТЗ."""
        Path(path).write_text(json.dumps({
            "max_spine_tilt_deg": self.rules["max_spine_tilt"],
            "vb_axis_limit_deg": self.rules["vb_axis_limit"],
            "methodology_tilt_deg": METHODOLOGY_TILT_DEG,
            "artifact_rule": ({"feature": self.rules["artifact_feature"],
                               "threshold": self.rules["artifact_threshold"]}
                              if self.rules["artifact_feature"] else None),
            "heads": {k: {"threshold": v["threshold"], "n_features": len(v["features"])}
                      for k, v in self.heads.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
