"""Разбор вердикта: проверки области с величинами и порогами — одна сборка на сервис и демо.

Правила объясняются самим числом против порога, головы — SHAP. SHAP считается только по
бустинговой половине `MeanOfTwo`, поэтому вклад помечен `partial`.
"""

from __future__ import annotations

import numpy as np

SPINE_EXPLAIN_HEADS = ("spine_coverage",)
HIP_EXPLAIN_HEADS = ("hip_positioning", "hip_roi")

TASK_NAMES = {
    "spine_coverage": "укладка позвоночника",
    "hip_positioning": "укладка бедра",
    "hip_roi": "область интереса бедра",
}


def head_contrib(model, name: str, feats: dict, top: int = 6):
    """Вероятность головы и SHAP-вклад признаков; вклад пуст, если TreeExplainer не поднялся."""
    from dxa_feat.names import human

    h = model.heads.get(name)
    if not h:
        return None, []
    cols = h["features"]
    x = np.array([[model._value(feats, c) for c in cols]], dtype=float)
    prob = float(h["model"].predict_proba(x)[0, 1])
    try:
        import shap

        sv = np.array(shap.TreeExplainer(h["model"].a).shap_values(x)).reshape(-1)
    except Exception:
        return prob, []
    order = np.argsort(-np.abs(sv))[:top]
    out = []
    for i in order:
        col = cols[i]
        base = col.rsplit("_", 1)[0] if col.rsplit("_", 1)[-1] in ("mean", "min", "max") else col
        out.append({"feature": human(base), "code": col,
                    "value": round(float(x[0, i]), 3),
                    "impact": round(float(sv[i]), 3)})
    return prob, out


def tilt_of(model, feats: dict) -> float:
    """Наклон оси, которым решает сервис; -1, если не измерен."""
    from dxa_service.quality import spine_tilt

    return spine_tilt(model._value(feats, "vb_axis_deg"),
                      model._value(feats, "col_tilt_deg"),
                      model._value(feats, "axis_deg_max"))


def _rule_checks(model, feats: dict) -> list[dict]:
    """Проверки позвоночника, которые решаются сравнением с порогом."""
    tilt = tilt_of(model, feats)
    limit = float(model.rules["vb_axis_limit"])
    out = [{"task": "ось позвоночника", "kind": "rule",
            "value": round(float(tilt), 2), "threshold": round(limit, 2),
            "unit": "°", "fired": bool(tilt > limit), "shap": []}]
    name, thr = model.rules["artifact_feature"], model.rules["artifact_threshold"]
    if name and thr is not None:
        val = model._value(feats, name.rsplit("_", 1)[0])
        out.append({"task": "посторонние предметы", "kind": "rule",
                    "value": round(float(val), 5), "threshold": round(float(thr), 5),
                    "unit": "", "fired": bool(val > float(thr)), "shap": []})
    return out


def checks(model, region: str, feats: dict) -> list[dict]:
    """Проверки области: правила, затем головы; годность сюда не входит — она в самом ответе."""
    if model is None or not getattr(model, "heads", None):
        return []
    spine = region == "spine"
    out = _rule_checks(model, feats) if spine else []
    for name in (SPINE_EXPLAIN_HEADS if spine else HIP_EXPLAIN_HEADS):
        prob, sh = head_contrib(model, name, feats)
        if prob is None:
            continue
        thr = float(model.heads[name]["threshold"])
        out.append({"task": TASK_NAMES.get(name, name), "kind": "head",
                    "prob": round(prob, 3), "threshold": round(thr, 3),
                    "fired": bool(prob >= thr), "partial": bool(sh), "shap": sh})
    return out
