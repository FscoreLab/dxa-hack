"""Анатомически нормированные измерения бедра и позвоночника.

Бедро — в системе координат оси диафиза, величины нормированы на его ширину.
Позвоночник — центральная линия: наклон (укладка) отдельно от кривизны (сколиоз).
"""

from __future__ import annotations

import cv2
import numpy as np


def _largest_component(mask: np.ndarray, row_fill: bool = True) -> np.ndarray:
    """Крупнейшая компонента; при row_fill — сплошная заливка по строкам.

    Маска бывает «скорлупой» по кортикальному слою; заливка восстанавливает тело кости.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    m = mask if n <= 1 else lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if not row_fill:
        return m
    out = np.zeros_like(m)
    for y in range(m.shape[0]):
        xs = np.nonzero(m[y])[0]
        if len(xs):
            out[y, xs.min():xs.max() + 1] = True
    return out


def shaft_axis(mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Ось диафиза: прямая по центрам строк в нижней части кости.

    Не PCA: обрезанная краем кадра кость бывает шире, чем выше, и главная ось
    ложится горизонтально.
    """
    m = _largest_component(mask)
    rows = np.nonzero(m.any(1))[0]
    if len(rows) < 20:
        return 0.0, np.array([0.0, 1.0])
    y_lo, y_hi = int(rows.min()), int(rows.max())
    cut = int(y_lo + 0.60 * (y_hi - y_lo))
    ys, cx = [], []
    for y in range(cut, y_hi + 1):
        xs = np.nonzero(m[y])[0]
        if len(xs):
            ys.append(y)
            cx.append(0.5 * (xs.min() + xs.max()))
    if len(ys) < 8:
        return 0.0, np.array([0.0, 1.0])
    k = float(np.polyfit(np.asarray(ys, float), np.asarray(cx, float), 1)[0])
    ang = float(np.degrees(np.arctan(k)))          # наклон к вертикали
    d = np.array([np.sin(np.radians(ang)), np.cos(np.radians(ang))])
    return ang, d


def _rotate(mask: np.ndarray, deg: float) -> np.ndarray:
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)
    return cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0


def femur_profile(mask: np.ndarray, side: str) -> dict:
    """Выступ малого вертела и геометрия проксимального отдела.

    Выступ — выход медиального контура за линию, экстраполированную с диафиза вверх.
    """
    ang, _ = shaft_axis(mask)
    m = _largest_component(_rotate(mask, ang))
    h, w = m.shape
    rows = np.arange(h)
    left = np.full(h, np.nan)
    right = np.full(h, np.nan)
    for y in rows:
        xs = np.nonzero(m[y])[0]
        if len(xs):
            left[y], right[y] = xs.min(), xs.max()
    ok = ~np.isnan(left)
    if ok.sum() < 20:
        return {"empty_femur": 1.0}

    width = right - left
    # зоны от протяжённости кости, а не кадра: в обрезанном кадре низ бывает пуст
    filled = rows[ok]
    y_lo, y_hi = float(filled.min()), float(filled.max())
    span = y_hi - y_lo + 1
    shaft_rows = rows[ok & (rows > y_lo + 0.70 * span)]
    if len(shaft_rows) < 5:
        return {"empty_femur": 1.0}
    shaft_w = float(np.nanmedian(width[shaft_rows]))

    # медиальная сторона: у правого бедра таз слева на изображении и наоборот
    medial_is_left = side == "left"
    med = left if medial_is_left else right
    sign = 1.0 if medial_is_left else -1.0   # выступ = уход медиального контура наружу

    p = np.polyfit(shaft_rows, med[shaft_rows], 1)
    prox = rows[ok & (rows > y_lo + 0.30 * span) & (rows < y_lo + 0.70 * span)]
    if len(prox) < 5:
        return {"empty_femur": 1.0}
    expected = np.polyval(p, prox)
    protr = sign * (expected - med[prox])          # положительное — контур вышел внутрь
    protr_max = float(np.nanmax(protr))
    return {
        "empty_femur": 0.0,
        "shaft_angle_deg": abs(ang),
        "shaft_width_px": shaft_w,
        "lt_protrusion_px": protr_max,
        # безразмерный индекс малого вертела — переносится между масштабами
        "lt_index": float(protr_max / shaft_w) if shaft_w > 0 else 0.0,
        "lt_protrusion_mean": float(np.nanmean(np.clip(protr, 0, None))),
        "prox_width_ratio": float(np.nanmax(width[prox]) / shaft_w) if shaft_w > 0 else 0.0,
        "width_profile_std": float(np.nanstd(width[ok]) / (shaft_w + 1e-6)),
    }


def spine_centerline(mask: np.ndarray) -> dict:
    """Наклон оси отдельно от кривизны плюс разброс наклона по уровням."""
    m = _largest_component(mask)
    h, w = m.shape
    cx = np.full(h, np.nan)
    width = np.full(h, np.nan)
    for y in range(h):
        xs = np.nonzero(m[y])[0]
        if len(xs):
            cx[y] = xs.mean()
            width[y] = xs.max() - xs.min()
    ok = ~np.isnan(cx)
    if ok.sum() < 40:
        return {"empty_spine": 1.0}

    ys = np.arange(h)[ok]
    p = np.polyfit(ys, cx[ok], 1)
    tilt = float(abs(np.degrees(np.arctan(p[0]))))
    resid = cx[ok] - np.polyval(p, ys)
    med_w = float(np.nanmedian(width[ok]))

    # наклон по третям: у сколиоза знак меняется, у перекошенной укладки нет
    thirds = np.array_split(np.arange(len(ys)), 3)
    slopes = []
    for t in thirds:
        if len(t) > 10:
            slopes.append(np.polyfit(ys[t], cx[ok][t], 1)[0])
    slopes = np.array(slopes) if slopes else np.zeros(1)

    return {
        "empty_spine": 0.0,
        "tilt_deg": tilt,
        "curv_rms_norm": float(np.sqrt((resid ** 2).mean()) / (med_w + 1e-6)),
        "curv_max_norm": float(np.abs(resid).max() / (med_w + 1e-6)),
        "slope_sign_change": float(np.ptp(np.sign(slopes)) > 0),
        "slope_spread": float(np.degrees(np.arctan(np.ptp(slopes)))),
        "width_cv": float(np.nanstd(width[ok]) / (med_w + 1e-6)),
    }


def femur_landmarks(mask: np.ndarray, side: str = "right") -> dict:
    """Форма проксимального отдела бедра по карте расстояний — прокси ротации.

    Имена нарочно не анатомические: головка на DXA скрыта впадиной, маска
    обрывается на шейке. Меряются вписанный круг в медиальной проксимальной
    части, перетяжка к диафизу и вынос большого вертела от оси.
    """
    m = _largest_component(mask)
    ang, d = shaft_axis(m)
    h, w = m.shape
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return {"empty_lm": 1.0}
    y_lo, y_hi = int(ys.min()), int(ys.max())
    span = max(1, y_hi - y_lo)

    # опора на диафизе: центр масс маски в нижних 20 %
    low = m.copy(); low[: int(y_lo + 0.80 * span)] = False
    lys, lxs = np.nonzero(low)
    if len(lxs) < 10:
        return {"empty_lm": 1.0}
    sy, sx = float(lys.mean()), float(lxs.mean())

    # глобальный максимум садится в более широкую межвертельную область,
    # поэтому ищем только медиально и в верхней части кости
    lateral = -1.0 if side == "right" else 1.0
    gy, gx = np.mgrid[0:h, 0:w]
    off_map = lateral * ((gx - sx) * d[1] - (gy - sy) * d[0])
    top = np.where((off_map < 0) & (gy < y_lo + 0.45 * span), dt, 0.0)
    if top.max() <= 0:
        top = np.where(gy < y_lo + 0.45 * span, dt, 0.0)
    hy, hx = np.unravel_index(int(np.argmax(top)), top.shape)
    prox_r = float(dt[hy, hx])
    if prox_r <= 0:
        return {"empty_lm": 1.0}

    # шейка: минимум расстояния на отрезке головка → диафиз, вне самой головки
    t = np.linspace(0.0, 1.0, 120)
    px = hx + (sx - hx) * t
    py = hy + (sy - hy) * t
    vals = dt[np.clip(py.astype(int), 0, h - 1), np.clip(px.astype(int), 0, w - 1)]
    seg = (t > 0.15) & (t < 0.75)
    k = int(np.argmin(np.where(seg, vals, np.inf)))
    ny, nx, neck_r = float(py[k]), float(px[k]), float(vals[k])

    # большой вертел: точка кости, максимально смещённая латерально от оси
    off = lateral * ((xs - sx) * d[1] - (ys - sy) * d[0])
    up = ys < y_lo + 0.55 * span
    gi = int(np.argmax(np.where(up, off, -np.inf))) if up.any() else int(np.argmax(off))
    gt_off = float(off[gi])

    shaft_w = 2.0 * float(np.median(dt[low]))
    neck_vec = np.array([hx - nx, hy - ny], float)
    nl = float(np.linalg.norm(neck_vec))
    cosa = float(np.dot(neck_vec / (nl + 1e-6), d))
    return {
        "empty_lm": 0.0,
        "prox_r_px": prox_r,
        # шейка короче и толще на проекции при ротации — оба признака безразмерны
        "prox_dist_ratio": nl / (2 * prox_r),
        "isthmus_ratio": neck_r / prox_r,
        "prox_axis_deg": float(np.degrees(np.arccos(np.clip(cosa, -1, 1)))),
        "gt_offset_ratio": gt_off / (2 * prox_r),
        "prox_shaft_ratio": (2 * prox_r) / (shaft_w + 1e-6),
        "_pts": {"prox": (hx, hy), "isthmus": (nx, ny), "gt": (float(xs[gi]), float(ys[gi])),
                 "shaft": (sx, sy)},
    }


def spine_discs(img: np.ndarray, mask: np.ndarray) -> dict:
    """Межпозвонковые диски: отклик шаблона «светлое — тёмное — светлое» плюс ДП.

    ДП со штрафом за отклонение шага от медианного: пропуск диска не сдвигает
    остальные, неравные тела не копят ошибку, как у равномерной решётки.
    """
    m = _largest_component(mask)
    h, w = m.shape
    cx = np.full(h, np.nan)
    half = np.zeros(h)
    for y in range(h):
        xs = np.nonzero(m[y])[0]
        if len(xs):
            cx[y] = 0.5 * (xs.min() + xs.max())
            half[y] = 0.5 * (xs.max() - xs.min())
    ok = ~np.isnan(cx)
    if ok.sum() < 40:
        return {"empty_disc": 1.0}
    ys = np.arange(h)[ok]
    y_lo, y_hi = int(ys.min()), int(ys.max())
    span = max(1, y_hi - y_lo)

    # яркость поперёк колонны: усредняем по ширине тела, а не по узкой полоске
    prof = np.zeros(h)
    for y in ys:
        # полоса узкая намеренно: шире в профиль попадают поперечные отростки
        c, r = int(cx[y]), max(3, int(0.40 * half[y]))
        prof[y] = img[y, max(0, c - r): min(w, c + r + 1)].mean()

    # шаблон диска: светлое — тёмное — светлое; ширина полосы около 1/6 позвонка
    p = prof[y_lo:y_hi + 1].astype(np.float32)
    p = cv2.blur(p.reshape(-1, 1), (1, 3)).ravel()
    best = None
    for vh in range(max(3, span // 40), max(5, span // 12)):
        k = np.concatenate([np.full(vh, 0.5), np.full(vh, -1.0), np.full(vh, 0.5)])
        k = k - k.mean()
        r = np.convolve(p, k[::-1], mode="same")
        if best is None or r.max() > best[0].max():
            best = (r, vh)
    resp, vh = best

    # шаг между дисками: автокорреляция отклика, диапазон задан анатомией
    q = resp - resp.mean()
    ac = np.correlate(q, q, "full")[len(q) - 1:]
    lo, hi = max(5, len(q) // 8), max(8, len(q) // 3)
    if hi <= lo + 1:
        return {"empty_disc": 1.0}
    step = float(np.argmax(ac[lo:hi]) + lo)

    # локальные максимумы без требования положительности: при низком контрасте все в минусе
    cand = [i for i in range(1, len(resp) - 1)
            if resp[i] >= resp[i - 1] and resp[i] > resp[i + 1]]
    if len(cand) < 2:
        return {"empty_disc": 1.0}

    # сумма откликов минус штраф за неравный шаг; без бонуса за узел
    # выигрывает короткая цепочка из пары сильных откликов
    scale = float(np.abs(resp).max() + 1e-6)
    LAM = 0.02 * scale
    BONUS = 0.45 * scale
    score = [resp[i] + BONUS for i in cand]
    prev = [-1] * len(cand)
    for j in range(len(cand)):
        for i in range(j):
            d = cand[j] - cand[i]
            if d < 0.55 * step or d > 1.8 * step:
                continue
            val = score[i] + resp[cand[j]] + BONUS - LAM * abs(d - step)
            if val > score[j]:
                score[j], prev[j] = val, i
    j = int(np.argmax(score))
    chain = []
    while j >= 0:
        chain.append(cand[j])
        j = prev[j]
    disc_y = np.array(sorted(y_lo + np.array(chain)))

    gaps = np.diff(disc_y) if len(disc_y) > 1 else np.array([step])
    return {
        "empty_disc": 0.0,
        "disc_n": float(len(disc_y)),
        "disc_step_norm": float(step / span),
        "disc_step_cv": float(np.std(gaps) / (np.mean(gaps) + 1e-6)),
        "disc_top_gap": float((disc_y.min() - y_lo) / step) if len(disc_y) else -1.0,
        "disc_bot_gap": float((y_hi - disc_y.max()) / step) if len(disc_y) else -1.0,
        "disc_strength": float(np.mean(resp[[c for c in chain]]) / (np.abs(resp).max() + 1e-6)),
        "_discs": disc_y.tolist(),
    }


def spine_segments(img: np.ndarray, mask: np.ndarray) -> dict:
    """Разбиение колонны на позвонки и охват в позвонках; в модель не идёт.

    Период — автокорреляцией профиля, устойчивее к пропущенному минимуму.
    """
    m = _largest_component(mask)
    h, w = m.shape
    cx = np.full(h, np.nan)
    for y in range(h):
        xs = np.nonzero(m[y])[0]
        if len(xs):
            cx[y] = 0.5 * (xs.min() + xs.max())
    ok = ~np.isnan(cx)
    if ok.sum() < 40:
        return {"empty_seg": 1.0}
    ys = np.arange(h)[ok]

    # профиль яркости по центральной полосе: тело позвонка светлее диска
    half = max(3, int(0.12 * np.nanmedian([np.count_nonzero(m[y]) for y in ys])))
    prof = np.full(h, np.nan)
    for y in ys:
        c = int(cx[y])
        prof[y] = img[y, max(0, c - half): min(w, c + half + 1)].mean()
    p = prof[ys]
    p = p - cv2.blur(p.reshape(-1, 1).astype(np.float32), (1, 25)).ravel()

    p = cv2.blur(p.reshape(-1, 1).astype(np.float32), (1, 5)).ravel()

    # диапазон периода задан анатомией (3–8 тел в поле), иначе argmax цепляется за первый лаг
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    lo, hi = max(5, len(p) // 8), max(8, len(p) // 3)
    if hi <= lo + 1:
        return {"empty_seg": 1.0}
    period = float(np.argmax(ac[lo:hi]) + lo)
    strength = float(ac[int(period)] / (ac[0] + 1e-6))

    # фаза гармоники усредняет весь профиль, в отличие от отдельных минимумов
    t = np.arange(len(p))
    z = np.sum(p * np.exp(-2j * np.pi * t / period))
    k0 = (float(np.angle(z)) / (2 * np.pi)) * period

    # фаза указывает на яркие тела; диски в противофазе — берём решётку, где профиль ниже
    def grid_at(k):
        return np.arange(k % period, len(p), period)

    cands = [grid_at(k0), grid_at(k0 + period / 2)]
    grid = min(cands, key=lambda g: p[np.clip(g.astype(int), 0, len(p) - 1)].mean()
               if len(g) else np.inf)
    # тела не равны по высоте: узел подтягивается к минимуму в пределах четверти периода
    win = max(2, int(0.25 * period))
    snapped = []
    for k in grid.astype(int):
        a0, b0 = max(0, k - win), min(len(p), k + win + 1)
        if b0 > a0:
            snapped.append(a0 + int(np.argmin(p[a0:b0])))
    disc_y = ys[np.clip(np.unique(snapped), 0, len(ys) - 1)] if snapped else np.array([])

    y_top, y_bot = float(ys.min()), float(ys.max())
    top_gap = (disc_y.min() - y_top) / period if len(disc_y) else np.nan
    bot_gap = (y_bot - disc_y.max()) / period if len(disc_y) else np.nan
    return {
        "empty_seg": 0.0,
        "vert_period_norm": period / (y_bot - y_top + 1),   # период в долях колонны
        "vert_strength": strength,
        "n_discs": float(len(disc_y)),
        "n_vert_span": (y_bot - y_top) / period,            # сколько позвонков в поле
        # сколько позвонка не хватает до края поля сверху и снизу
        "top_gap_vert": float(top_gap) if top_gap == top_gap else -1.0,
        "bot_gap_vert": float(bot_gap) if bot_gap == bot_gap else -1.0,
        "_discs": disc_y.tolist(),
    }


def medial_proximal_contrast(img: np.ndarray, mask: np.ndarray) -> dict:
    """Контраст медиальной проксимальной зоны бедра — прокси качества укладки.

    Трактовка как «видимость малого вертела» не подтвердилась; низкий контраст
    зоны — у дефектных, вопреки ожиданию по ISCD.
    """
    m = _largest_component(mask)
    ang, d = shaft_axis(m)
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return {"med_prox_contrast": -1.0, "med_prox_bright_frac": -1.0}
    y_lo, y_hi = float(ys.min()), float(ys.max())
    span = max(1.0, y_hi - y_lo)

    low = m.copy(); low[: int(y_lo + 0.80 * span)] = False
    lys, lxs = np.nonzero(low)
    if len(lxs) < 10:
        return {"med_prox_contrast": -1.0, "med_prox_bright_frac": -1.0}
    sy, sx = float(lys.mean()), float(lxs.mean())

    # медиальная сторона (маска приведена к правому бедру: медиаль справа),
    # зона между вертельной областью и верхом диафиза
    gy, gx = np.mgrid[0:h, 0:w]
    off = (gx - sx) * d[1] - (gy - sy) * d[0]        # >0 — медиально
    zone = m & (off > 0) & (gy > y_lo + 0.35 * span) & (gy < y_lo + 0.72 * span)
    if zone.sum() < 20:
        return {"med_prox_contrast": -1.0, "med_prox_bright_frac": -1.0, "med_prox_ok": 0.0}

    # Заглушка отказа (-1) обязана отличаться от измеренного нуля, иначе модель
    # учит «не посчиталось» (эндопротезы, почти все дефектные) вместо вертела.
    base = float(np.median(img[m]))
    scale = float(np.std(img[m])) + 1e-6
    z = (img[zone] - base) / scale
    return {
        "med_prox_contrast": float(np.percentile(z, 90)),          # насколько ярче кости пик зоны
        "med_prox_bright_frac": float((z > 1.0).mean()),          # какая доля зоны заметно ярче
        "med_prox_ok": 1.0,
    }


def _soft_bone(img: np.ndarray) -> np.ndarray:
    """Кость мягким порогом (плотнее мягких тканей) — где `bone_mask` теряет крыло таза."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    try:
        from skimage.filters import threshold_multiotsu

        m = (u8 > threshold_multiotsu(u8, classes=3)[0]).astype(np.uint8)
    except Exception:
        return np.zeros(u8.shape, dtype=bool)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return m > 0


def vertebra_levels(img: np.ndarray, mask: np.ndarray, discs: list[float]) -> dict:
    """Нумерация дисков от гребней подвздошных костей.

    Линия Якоби (верх гребней) проходит на уровне L4–L5: ближайший диск — L4/L5,
    выше L3/L4 и т. д. Рёбра как второй якорь непригодны: в поле попадают редко.
    """
    # гребни — отдельные компоненты, поэтому ищем их по полной маске, а не по колонне
    col = _largest_component(mask)
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    cx = []
    for y in range(int(0.35 * h), int(0.65 * h)):
        xs = np.nonzero(col[y])[0]
        if len(xs):
            cx.append(np.median(xs))
    if not cx or not discs:
        return {"no_levels": 1.0}
    c = int(np.median(cx))
    half = max(4, int(0.16 * w))

    # костная масса ВНЕ колонны по строкам: гребни дают резкий максимум внизу
    def _side_profile(src):
        side = np.asarray(src, dtype=bool).copy()
        side[:, max(0, c - half): min(w, c + half)] = False
        return side.sum(1).astype(float)

    prof = _side_profile(m)
    lo = int(0.5 * h)
    # Тусклое крыло таза Otsu относит к мягким тканям. Мягкий порог — только
    # запасной: на обычных кадрах он сдвигает якорь вверх и ломает нумерацию.
    if prof[lo:].max() < 3:
        prof = _side_profile(_soft_bone(img))
    if prof[lo:].max() < 3:
        return {"no_levels": 1.0}
    # линия Якоби — по верху гребней, а максимум боковой массы ниже: берём
    # первое уверенное появление боковой кости сверху
    thr = max(2.0, 0.15 * float(prof[lo:].max()))
    above = np.nonzero(prof[lo:] >= thr)[0]
    if len(above) == 0:
        return {"no_levels": 1.0}
    crest_y = float(lo + int(above[0]))

    d = np.asarray(sorted(discs), dtype=float)
    k = int(np.argmin(np.abs(d - crest_y)))      # этот диск и есть L4/L5

    # выше L4/L5 в поле максимум четыре диска; остальное — грудной отдел или ложные
    MAX_ABOVE = 4
    if k > MAX_ABOVE:
        extra = k - MAX_ABOVE
        d = d[extra:]
        k = MAX_ABOVE
    d = d[: k + 3]                                # ниже L4/L5 держим L5/S1 и запас
    # имена вверх от якоря: L4/L5, L3/L4, L2/L3, L1/L2, Th12/L1
    names = {}
    for i in range(len(d)):
        step = k - i
        names[i] = 4 - step                       # 4 → L4/L5, 3 → L3/L4 …
    top_level = names[0]
    ys, _ = np.nonzero(col)
    y_top = float(ys.min())
    gap = (d[0] - y_top) / max(np.median(np.diff(d)) if len(d) > 1 else 1.0, 1.0)

    return {
        "no_levels": 0.0,
        "crest_rel": crest_y / h,
        "levels_dropped": float(max(0, len(discs) - len(d))),
        # какой позвонок виден сверху: 1 — L1 в кадре, 0 и меньше — поле выше
        "top_vertebra": float(top_level),
        # виден ли L1 целиком: над верхним диском есть целое тело
        "l1_visible": float(top_level <= 1 and gap > 0.7),
        # гребни в кадре — прямое требование методички
        "crest_in_frame": float(crest_y < 0.97 * h),
        "levels_below_crest": float(len(d) - k - 1),
    }


def _body_centers(img: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центры тел позвонков построчно: центроид строки, взвешенный по яркости.

    Тело плотнее отростков и перевешивает их. Жёсткая отсечка по яркости хуже:
    отростки в проекции накладываются на тело.
    """
    h, w = col.shape
    cx, half = np.full(h, np.nan), np.full(h, np.nan)
    for y in range(h):
        xs = np.nonzero(col[y])[0]
        if not len(xs):
            continue
        wts = img[y, xs].astype(np.float64)
        wts = wts - wts.min() + 1e-6
        cx[y] = float((xs * wts).sum() / wts.sum())
        half[y] = 0.5 * (xs.max() - xs.min())
    return cx, half


def spine_column(img: np.ndarray, mask: np.ndarray, discs: list[float] | None = None) -> dict:
    """Признаки по колонне позвонков без таза и рёбер.

    Колонна — центральная связная полоса от верхнего диска до уровня гребней;
    правило 5° применяется к её наклону.
    """
    m = np.asarray(mask, dtype=bool)
    col = _largest_component(m)
    h, w = col.shape
    # угол от вертикали кадра — это ось сканирования
    cx, half = _body_centers(img, col)
    ok = ~np.isnan(cx)
    if ok.sum() < 40:
        return {"empty_column": 1.0}
    rows = np.arange(h)[ok]
    y_lo, y_hi = int(rows.min()), int(rows.max())

    # низ колонны — уровень гребней: там боковая костная масса резко растёт
    side = m.copy()
    c_med = int(np.nanmedian(cx))
    band = max(6, int(0.18 * w))
    side[:, max(0, c_med - band): min(w, c_med + band)] = False
    prof = side.sum(1).astype(float)
    lo = int(0.5 * h)
    bottom = y_hi
    if prof[lo:].max() >= 3:
        thr = max(2.0, 0.15 * float(prof[lo:].max()))
        above = np.nonzero(prof[lo:] >= thr)[0]
        if len(above):
            bottom = min(y_hi, lo + int(above[0]))

    top = y_lo
    if discs:
        d = [v for v in sorted(discs) if y_lo <= v <= bottom]
        if d:
            top = int(max(y_lo, d[0] - (d[1] - d[0] if len(d) > 1 else 20)))

    seg = rows[(rows >= top) & (rows <= bottom)]
    if len(seg) < 25:
        return {"empty_column": 1.0}

    # Наклон — по центрам крайних тел, как в методике: искривление без наклона
    # не нарушение, а регрессия по всей колонне их смешивает.
    k = max(3, int(0.12 * len(seg)))
    top_y, top_x = float(seg[:k].mean()), float(np.nanmean(cx[seg[:k]]))
    bot_y, bot_x = float(seg[-k:].mean()), float(np.nanmean(cx[seg[-k:]]))
    tilt = float(abs(np.degrees(np.arctan((bot_x - top_x) / max(bot_y - top_y, 1e-6)))))
    # Излом между половинами: кривая укладка наклоняет колонну целиком,
    # сколиоз гнёт дугой — половины расходятся.
    _h = len(seg) // 2
    def _ang(sy, vx):
        return float(np.degrees(np.arctan(np.polyfit(sy.astype(float), vx, 1)[0])))
    kink = (abs(_ang(seg[:_h], cx[seg[:_h]]) - _ang(seg[_h:], cx[seg[_h:]]))
            if _h >= 5 else 0.0)
    # кривизна — от регрессии: это мера отклонения от прямой
    p = np.polyfit(seg.astype(float), cx[seg], 1)
    resid = cx[seg] - np.polyval(p, seg)
    med_w = float(np.nanmedian(half[seg])) * 2.0

    colmask = np.zeros_like(col)
    for y in seg:
        xs = np.nonzero(col[y])[0]
        if len(xs):
            colmask[y, xs.min():xs.max() + 1] = True
    area = float(colmask.sum())

    return {
        "empty_column": 0.0,
        "col_tilt_deg": tilt,                       # к нему и применяется правило 5°
        "col_curv_norm": float(np.sqrt((resid ** 2).mean()) / (med_w + 1e-6)),
        "col_kink_deg": kink,                       # расхождение половин оси
        "col_width_med": med_w / w,                 # ширина колонны в долях кадра
        "col_width_cv": float(np.nanstd(half[seg] * 2) / (med_w + 1e-6)),
        "col_height_frac": (bottom - top) / h,      # сколько кадра занимает колонна
        "col_area_frac": area / (h * w),
        "col_top_margin": top / h,                  # где начинается колонна
        "col_bottom_margin": (h - bottom) / h,      # запас под гребнями
        "col_crest_found": float(bottom < y_hi),
    }


def proximal_grid(img: np.ndarray, mask: np.ndarray, ny: int = 4, nx: int = 3) -> dict:
    """Проксимальный отдел бедра сеткой в координатах диафиза, нормированных на его ширину.

    В каждой ячейке: доля кости и средняя яркость относительно медианы кости.
    """
    m = _largest_component(mask)
    ang, d = shaft_axis(m)
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return {"empty_grid": 1.0}
    y_lo, y_hi = float(ys.min()), float(ys.max())
    span = max(1.0, y_hi - y_lo)

    low = m.copy(); low[: int(y_lo + 0.80 * span)] = False
    lys, lxs = np.nonzero(low)
    if len(lxs) < 10:
        return {"empty_grid": 1.0}
    sy, sx = float(lys.mean()), float(lxs.mean())
    shaft_w = max(4.0, 2.0 * float(np.median(cv2.distanceTransform(m.astype(np.uint8),
                                                                   cv2.DIST_L2, 5)[low])))

    gy, gx = np.mgrid[0:h, 0:w]
    # вдоль оси: 0 у верха кости, 1 у опоры на диафизе; поперёк: в ширинах диафиза
    along = ((gy - y_lo) * d[1] + (gx - sx) * d[0]) / span
    across = ((gx - sx) * d[1] - (gy - sy) * d[0]) / shaft_w

    base = float(np.median(img[m]))
    scale = float(np.std(img[m])) + 1e-6
    out = {"empty_grid": 0.0}
    edges_y = np.linspace(0.0, 0.75, ny + 1)          # только проксимальные три четверти
    edges_x = np.linspace(-1.6, 1.6, nx + 1)
    for i in range(ny):
        for j in range(nx):
            cell = ((along >= edges_y[i]) & (along < edges_y[i + 1])
                    & (across >= edges_x[j]) & (across < edges_x[j + 1]))
            n = int(cell.sum())
            if n < 12:
                out[f"g{i}{j}_fill"] = -1.0
                out[f"g{i}{j}_int"] = -1.0
                continue
            inside = cell & m
            out[f"g{i}{j}_fill"] = float(inside.sum()) / n
            out[f"g{i}{j}_int"] = (float(img[inside].mean()) - base) / scale if inside.any() else -1.0
    return out
