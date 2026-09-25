"""Сквозной пайплайн: DICOM на входе — строка отчёта ТЗ на каждое изображение.

Ошибка на файле или неуверенность в области дают строку со статусом Failure, пакет не падает.
"""

from __future__ import annotations

import os
import time

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from dxa_io import read
from dxa_feat import registry
from dxa_feat.context import FrameContext
from dxa_seg.region import bone_mask
from dxa_service.embed import components, embedding
from dxa_service.femur_mask import femur_mask

# Коды нарушений — по ТЗ (6 штук), а не по склеенным флагам разметки.
VIOLATIONS = ("spine_coverage", "spine_axis", "spine_artifact",
              "hip_positioning", "hip_rotation", "hip_roi_coverage")

# Значения заданы ТЗ дословно; сторона бедра наружу не выходит.
SPINE_NAME = "Поясничный отдел позвоночника"
FEMUR_NAME = "Проксимальный отдел бедра"
REGION_NAMES = {"spine": SPINE_NAME, "hip_right": FEMUR_NAME,
                "hip_left": FEMUR_NAME, "unknown": ""}   # у Failure — пусто, вне списка организаторов нет значений

# Тексты нарушений заданы ТЗ дословно: метрика сверяет строки, менять нельзя.
VIOLATION_NAMES = {
    "spine_coverage": "Некорректная укладка",
    "spine_axis": "Не выровнена ось позвоночника",
    "spine_artifact": "Присутствуют посторонние предметы",
    "hip_positioning": "Некорректная укладка",
    "hip_rotation": "Некорректная укладка",
    "hip_roi_coverage": "Некорректная область интереса",
}


def violation_text(codes: list[str]) -> str:
    """Внутренние коды → тексты ТЗ через «;» без повторов (у разных кодов бывает один текст)."""
    out: list[str] = []
    for c in codes:
        t = VIOLATION_NAMES.get(c)
        if t and t not in out:
            out.append(t)
    return ";".join(out)


@dataclass
class Row:
    path_to_study: str
    study_uid: str
    image_uid: str
    anatomical_region: str
    quality_class: int | str
    violation_type: str
    processing_status: str
    time_of_processing: float
    # Сверх контракта: непрерывная оценка для ROC-AUC.
    quality_prob: float = 0.0
    # Служебное: сторона бедра, по anatomical_region её не восстановить. Вырезается перед выдачей.
    region_key: str = ""
    # Сверх контракта: пояснение вердикта для врача.
    details: str = ""


def region_features(img: np.ndarray) -> dict:
    """Признаки формы для классификатора области."""
    m = bone_mask(img)
    ys, xs = np.nonzero(m)
    h, w = img.shape
    if len(xs) < 20:
        return {k: 0.0 for k in ("aspect", "span", "fill", "area", "period",
                                 "top_heavy", "bot_heavy", "row_var")}
    bh = ys.max() - ys.min() + 1
    bw = xs.max() - xs.min() + 1
    prof = (img * m).sum(axis=1)
    p = prof - prof.mean()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = max(3, int(0.08 * len(p))), max(5, int(0.45 * len(p)))
    row = m.sum(1).astype(float)
    return {
        "aspect": float(bh / bw),
        "span": float(bw / w),
        "fill": float(m.sum() / (bh * bw)),
        "area": float(m.mean()),
        "period": float(ac[lo:hi].max()) if hi > lo else 0.0,
        "top_heavy": float(m[: h // 3].sum() / max(m.sum(), 1)),
        "bot_heavy": float(m[2 * h // 3:].sum() / max(m.sum(), 1)),
        "row_var": float(row.std() / (row.mean() + 1e-9)),
    }


_REGION_CLF = None
_REGION_CLF_EMB = None


def _load_region_clf(kind: str = "emb"):
    """Классификатор области: основной на эмбеддинге, запасной на силуэте — на случай недоступного энкодера."""
    global _REGION_CLF, _REGION_CLF_EMB
    name = "region_clf_emb.pkl" if kind == "emb" else "region_clf.pkl"
    cache = _REGION_CLF_EMB if kind == "emb" else _REGION_CLF
    if cache is None:
        import pickle
        p = Path(__file__).resolve().parents[1] / "models" / name
        cache = pickle.load(open(p, "rb")) if p.exists() else False
        if kind == "emb":
            _REGION_CLF_EMB = cache
        else:
            _REGION_CLF = cache
    return cache


def classify_region(img: np.ndarray) -> tuple[str, str, float]:
    """(регион, сторона, уверенность) по изображению: тегов области в данных нет."""
    m = bone_mask(img)
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return "unknown", "", 0.0


    prof = (img * m).sum(axis=1)
    p = prof - prof.mean()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = max(3, int(0.08 * len(p))), max(5, int(0.45 * len(p)))
    strength = float(ac[lo:hi].max()) if hi > lo else 0.0

    bh = ys.max() - ys.min() + 1
    bw = xs.max() - xs.min() + 1
    aspect = bh / bw

    clf_emb = _load_region_clf("emb")
    vec = embedding(img, m) if clf_emb else None
    clf = _load_region_clf("feat")
    if clf_emb and vec is not None:
        proba = clf_emb["model"].predict_proba(vec.reshape(1, -1))[0]
        classes = list(clf_emb["model"].classes_)
        is_spine = classes[int(np.argmax(proba))] == "spine"
        conf = float(proba.max())
    elif clf:
        f = region_features(img)
        x = np.array([[f[c] for c in clf["features"]]])
        proba = clf["model"].predict_proba(x)[0]
        classes = list(clf["model"].classes_)
        is_spine = classes[int(np.argmax(proba))] == "spine"
        conf = float(proba.max())
    else:
        # Эвристика без обученной модели; score — «похожесть на позвоночник», для бедра уверенность 1 - score.
        score = 0.6 * min(strength / 0.35, 1.0) + 0.4 * min(aspect / 1.6, 1.0)
        is_spine = score > 0.62
        conf = min(score, 1.0) if is_spine else min(1.0 - score, 1.0)
    if conf < 0.6:
        return "unknown", "", conf
    if is_spine:
        return "spine", "", conf

    top = xs[ys < np.percentile(ys, 30)]
    bot = xs[ys > np.percentile(ys, 70)]
    if len(top) == 0 or len(bot) == 0:
        return "unknown", "", 0.0
    side = "right" if top.mean() > bot.mean() else "left"
    return f"hip_{side}", side, round(conf, 2)


class _Meta:
    """Минимальный носитель масштаба: реестру от meta нужны только эти два поля."""

    def __init__(self, mm_x, mm_y):
        self.mm_per_px_x, self.mm_per_px_y = mm_x, mm_y


def measure(img: np.ndarray, region: str, side: str, mm_x, mm_y,
            fold: int | None = None, mask: np.ndarray | None = None) -> dict:
    """Признаки кадра из того же реестра, что и в обучении; маска бедра — от U-Net."""
    # fold — только для обучающих данных: маску даёт сеть, не видевшая фолда. В бою — ансамбль.
    # mask подаёт снаружи вложенная валидация, где U-Net переобучается внутри фолда.
    fem = mask if mask is not None else (
        None if region == "spine" else femur_mask(img, side, fold=fold))
    ctx = FrameContext(img=img, mask=fem, meta=_Meta(mm_x, mm_y), side=side,
                       region=region, fold=fold)
    f = registry.collect(ctx, region)
    # Эмбеддинг идёт мимо реестра: в обучении он берётся из заранее посчитанного
    # npz, здесь — считается энкодером на лету. Колонки одни и те же.
    f.update(components(embedding(ctx.img_r, ctx.mask_r)))
    return f


def analyse_file(path: str | Path, model=None,
                 with_frame: bool = False) -> tuple[Row, dict | None]:
    """Строка отчёта и, при with_frame, кадр, маска и признаки — тем же проходом, что и вердикт."""
    t0 = time.perf_counter()
    path = Path(path)
    try:
        meta, img = read(path)
        region, side, conf = classify_region(img)
        if region == "unknown":
            return Row(str(path.parent), meta.study_uid_tag, meta.sop_uid,
                       REGION_NAMES["unknown"], "", "", "Failure",
                       round(time.perf_counter() - t0, 3),
                       details="анатомическая область не определена"), None
        feats = measure(img, region, side, meta.mm_per_px_x, meta.mm_per_px_y)
        by_region = {region: _aggregate([feats])}
        _add_pairs(by_region)
        agg = by_region[region]
        q, vt, det, p = verdict_for(model, region, agg)
        row = Row(str(path.parent), meta.study_uid_tag, meta.sop_uid,
                  REGION_NAMES[region], q, vt, "Success",
                  round(time.perf_counter() - t0, 3),
                  details=det, quality_prob=round(float(p), 5), region_key=region)
        frame = None
        if with_frame:
            frame = {"img": img, "region": region, "side": side, "feats": agg,
                     "mask": bone_mask(img) if region == "spine"
                             else femur_mask(img, side)}
        return row, frame
    except Exception as exc:  # файл не должен ронять пакетную обработку
        # ТЗ п. 2.5: статус ровно Success / Failure, причина — в details
        return Row(str(path.parent), "", "", REGION_NAMES["unknown"], "", "", "Failure",
                   round(time.perf_counter() - t0, 3),
                   details=f"{type(exc).__name__}: {str(exc)[:120]}"), None


def process_file(path: str | Path, model=None) -> Row:
    """Один файл; исключений наружу не бросает."""
    return analyse_file(path, model)[0]


def verdict_for(model, region: str, feats: dict) -> tuple[int, str, str, float]:
    """Вердикт, типы, пояснение и оценка по агрегату области — единственное место решения."""
    if model is None:
        return 0, "", _explain(region, feats), 0.0
    scope = "spine" if region == "spine" else "hip"
    v = model.predict(scope, feats)
    # Годность — model.quality, а не bool(типы): на бедре у неё своя голова.
    q = model.quality(scope, feats, v)
    if scope == "hip":
        v = model.reconcile_hip(feats, v, q)
    p = model.quality_proba(scope, feats, v)
    return int(q), violation_text(v), _explain(region, feats), float(p)


# Обязан совпадать с PAIRED в build_features.py, иначе головы получат -1 по парным колонкам.
PAIRED = ("shaft_angle_deg", "med_prox_contrast", "width_profile_std",
          "area_frac", "prox_shaft_ratio")

WORST_MIN_PREFIX = ("margin_",)
WORST_MIN = ("contrast", "iliac_bottom_frac", "ribs_top_frac", "n_vertebrae")
WORST_MAX_PREFIX = ("touch_", "thin_")
WORST_MAX = ("axis_deg", "curvature", "saturated_frac", "n_components", "outside_bright")


def _aggregate(frames: list[dict]) -> dict:
    """Кадры области исследования — в одну строку; кроме среднего берётся худший случай по смыслу."""
    out = {}
    keys = set().union(*(f.keys() for f in frames)) if frames else set()
    for k in keys:
        vals = [float(f[k]) for f in frames if k in f and isinstance(f[k], (int, float))]
        if not vals:
            continue
        out[f"{k}_mean"] = float(np.mean(vals))
        if k.startswith(WORST_MIN_PREFIX) or k in WORST_MIN:
            out[f"{k}_min"] = float(np.min(vals))
        elif k.startswith(WORST_MAX_PREFIX) or k in WORST_MAX:
            out[f"{k}_max"] = float(np.max(vals))
    out["n_frames"] = float(len(frames))
    return out


def _add_pairs(by_region: dict[str, dict]) -> None:
    """Разность со второй стороной: убирает анатомию пациента, оставляет укладку."""
    opposite = {"hip_right": "hip_left", "hip_left": "hip_right"}
    for region, feats in by_region.items():
        other = by_region.get(opposite.get(region, ""))
        feats["has_pair"] = 1.0 if other else 0.0
        for base in PAIRED:
            col = f"{base}_mean"
            a = feats.get(col)
            b = other.get(col) if other else None
            if a is None or b is None:
                feats[f"{col}_diff"] = 0.0
                feats[f"{col}_asym"] = 0.0
            else:
                feats[f"{col}_diff"] = float(a - b)
                feats[f"{col}_asym"] = float(abs(a - b))


# Пометка только в details: сколиоз нарушением укладки не считается и в вердикт не идёт.
SCOLIOSIS_HINT = "признаки сколиоза (нарушением укладки не считается)"
# Порог без отложенной оценки — допустимо, пока пометка не влияет на вердикт.
SCOLIOSIS_CURV = 0.128


def _hints(region: str, f: dict) -> list[str]:
    """Подсказки врачу вне вердикта; подавлять ими правило оси нельзя — оно от этого хуже."""
    out = []
    if region == "spine":
        curv = f.get("curv_rms_norm_mean", f.get("curv_rms_norm", -1.0))
        if curv is not None and curv >= SCOLIOSIS_CURV:
            out.append(SCOLIOSIS_HINT)
    return out


def _explain(region: str, f: dict) -> str:
    """Короткое пояснение вердикта на языке критериев из методических рекомендаций."""
    LV = {5: "L5/S1", 4: "L4/L5", 3: "L3/L4", 2: "L2/L3", 1: "L1/L2", 0: "Th12/L1"}
    out = []
    if region == "spine":
        # Угол — из общего quality.spine_tilt, чтобы текст совпадал с проверкой.
        from dxa_service.quality import spine_tilt

        tilt = spine_tilt(f.get("vb_axis_deg_mean", f.get("vb_axis_deg", -1.0)),
                          f.get("col_tilt_deg_mean", f.get("col_tilt_deg", -1.0)),
                          f.get("axis_deg_max", f.get("axis_deg", -1.0)))
        if tilt >= 0:
            # Называем критерий методички, а срабатываем по своему порогу: наш угол ниже врачебного.
            from dxa_service.model import MAX_SPINE_TILT_DEG, METHODOLOGY_TILT_DEG
            ok = tilt <= MAX_SPINE_TILT_DEG
            out.append(f"наклон оси {tilt:.1f}°" + (
                f" (в пределах нормы, критерий {METHODOLOGY_TILT_DEG:.0f}°)" if ok
                else f" — превышает норму (критерий {METHODOLOGY_TILT_DEG:.0f}°)"))
        top = f.get("top_vertebra_mean", f.get("top_vertebra"))
        if top is not None and top > -9:
            out.append("сверху виден " + LV.get(int(round(top)), f"уровень {int(round(top))}"))
        # При расхождении верим крылу таза от детектора, а не яркостным гребням.
        wing = f.get("kp_wing_absent_mean", f.get("kp_wing_absent"))
        crest = f.get("crest_in_frame_mean", f.get("crest_in_frame"))
        if wing is not None and wing > 0.5:
            out.append("крыльев таза в кадре нет")
        elif crest is not None:
            out.append("гребни в кадре" if crest > 0.5 else "гребни не видны")
        from dxa_service.model import ARTIFACT_RULE
        if ARTIFACT_RULE:
            name, thr = ARTIFACT_RULE
            arcs = f.get(name, f.get(name.rsplit("_", 1)[0], 0.0))
            if arcs and arcs >= thr:
                out.append("посторонние тонкие структуры вне позвоночника")
    else:
        ang = f.get("shaft_angle_deg_mean", f.get("shaft_angle_deg", -1.0))
        if ang >= 0:
            out.append(f"наклон диафиза {ang:.1f}°")
        pair = f.get("has_pair", 0.0)
        asym = f.get("shaft_angle_deg_mean_asym", 0.0)
        if pair > 0.5 and asym > 4:
            out.append(f"стороны различаются на {asym:.1f}°")
        sat = f.get("saturated_frac_mean", f.get("saturated_frac", 0.0))
        if sat and sat > 0.02:
            out.append("яркие включения, возможен эндопротез")
    out += _hints(region, f)
    return "; ".join(out)


def _overlay_png(path: Path, region: str, feats: dict, model) -> bytes | None:
    """PNG-разбор изображения тем же draw_case, что у /explain/file."""
    try:
        import io

        from PIL import Image

        from dxa_service.explain import tilt_of
        from dxa_service.overlay import draw_case

        _, img = read(path)
        side = region.split("_")[1] if region.startswith("hip_") else ""
        tilt = tilt_of(model, feats) if region == "spine" and model is not None else None
        vis = draw_case(img, "spine" if region == "spine" else "hip", side,
                        tilt_deg=tilt, label_px=26)
        buf = io.BytesIO()
        Image.fromarray(np.asarray(vis)).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # картинка — дополнение: её сбой не меняет строку отчёта
        return None


def _unpack(root: Path):
    """Каталог или zip-архив (ТЗ п. 2.7): архивы распаковываются во временный каталог.

    Возвращает (временный каталог, {распакованный путь: путь архива}, каталоги для обхода).
    """
    import tempfile
    import zipfile

    tmp = tempfile.TemporaryDirectory(prefix="dxa_zip_")
    archives = ([root] if root.is_file() else
                sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"))
    dirs, shown = ([] if root.is_file() else [root]), {}
    for i, z in enumerate(archives):
        dst = Path(tmp.name) / str(i)
        try:
            with zipfile.ZipFile(z) as zf:
                for info in zf.infolist():
                    # архивы из русской Windows пишут имена в cp866 без UTF-8-флага,
                    # zipfile читает их как cp437 — отсюда кракозябры в path_to_study
                    if not info.flag_bits & 0x800:
                        try:
                            info.filename = info.filename.encode("cp437").decode("cp866")
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            pass
                    zf.extract(info, dst)    # extract сам отбрасывает «..» и абсолютные пути
        except zipfile.BadZipFile:
            continue
        dirs.append(dst)
        shown[str(dst)] = str(z)
    return tmp, shown, dirs


def process_dir(root: str | Path, model=None,
                overlays: dict[str, bytes] | None = None) -> list[dict]:
    """Каталог: вердикт на уровне «исследование × область», как учились головы, раздаётся всем кадрам области.

    `overlays` заполняется PNG-разборами по изображениям — дополнительные серии, ТЗ п. 2.7.
    """
    # DICOM часто лежит без расширения.
    tmp, shown, dirs = _unpack(Path(root))
    files = sorted(p for d in dirs for p in d.rglob("*")
                   if p.is_file() and p.suffix.lower() in ("", ".dcm", ".dicom"))
    rows = [process_file(f, None) for f in files]

    # Под try: сбой на одном файле не должен терять отчёт по каталогу.
    bank: dict[tuple[str, str], list[dict]] = {}
    for row, path in zip(rows, files):
        if row.processing_status != "Success":
            continue
        try:
            meta, img = read(path)
            region = row.region_key
            side = region.split("_")[1] if region.startswith("hip_") else ""
            t1 = time.perf_counter()
            feats = measure(img, "spine" if region == "spine" else "hip", side,
                            meta.mm_per_px_x, meta.mm_per_px_y)
            row.time_of_processing = round(row.time_of_processing
                                           + time.perf_counter() - t1, 3)
            bank.setdefault((row.study_uid, region), []).append(feats)
        except Exception as exc:
            row.processing_status = "Failure"
            row.details = f"{type(exc).__name__}: {str(exc)[:120]}"
            row.quality_class, row.violation_type = "", ''

    agg: dict[str, dict[str, dict]] = {}
    for (study, region), frames in bank.items():
        agg.setdefault(study, {})[region] = _aggregate(frames)
    for study, by_region in agg.items():
        _add_pairs(by_region)

    verdict: dict[tuple[str, str], tuple[int, str, str, float]] = {}
    for study, by_region in agg.items():
        for region, feats in by_region.items():
            verdict[(study, region)] = verdict_for(model, region, feats)

    out = []
    for row, path in zip(rows, files):
        if row.processing_status == "Success":
            q, vt, det, p = verdict.get((row.study_uid, row.region_key), (0, "", "", 0.0))
            row.quality_class, row.violation_type, row.details = q, vt, det
            row.quality_prob = round(float(p), 5)
            if overlays is not None:
                png = _overlay_png(path, row.region_key,
                                   agg.get(row.study_uid, {}).get(row.region_key, {}), model)
                if png:
                    overlays[f"{row.study_uid or 'study'}/{row.image_uid or path.stem}.png"] = png
        for src, name in shown.items():     # путь внутри архива, а не во временном каталоге
            if row.path_to_study.startswith(src):
                row.path_to_study = name + row.path_to_study[len(src):]
        d = asdict(row)
        d.pop("region_key", None)          # служебное поле в отчёт не идёт
        out.append(d)
    tmp.cleanup()
    return out
