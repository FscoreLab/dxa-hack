"""Отрисовка ориентиров, которыми решают признаки, поверх снимка.

Имена точек бедра нарочно не анатомические, как в femur_landmarks: «верх кости» на DXA — не головка.
"""

from __future__ import annotations

import cv2
import numpy as np

# Подписи через PIL: cv2.putText не умеет кириллицу.
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

SPINE = (90, 200, 255)      # уровни дисков
AXIS = (80, 120, 255)       # ось позвоночника
BONE = (255, 170, 60)       # контур кости
LM = (120, 255, 140)        # точки бедра
WING = (255, 220, 60)       # верх крыльев таза
CORRIDOR = (120, 170, 255)  # коридор колонны
ART = (255, 70, 70)         # посторонние предметы
MEDIAL = (255, 140, 200)    # малый вертел и контур диафиза


def _base(img: np.ndarray) -> np.ndarray:
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return np.stack([u8] * 3, -1).copy()


def _labels(vis: np.ndarray, items: list[tuple[str, int, int]],
            px: int = 13, color: tuple[int, int, int] = LM) -> np.ndarray:
    """Подписи кириллицей через PIL; размер px задаёт вызывающий под свой масштаб."""
    if not items:
        return vis
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(_FONT, px)
    except Exception as exc:
        # Без fonts-dejavu-core в образе пропадают все подписи — предупреждаем.
        import warnings
        warnings.warn(f"шрифт подписей недоступен ({exc}); картинка без текста",
                      RuntimeWarning, stacklevel=2)
        return vis
    im = Image.fromarray(vis)
    d = ImageDraw.Draw(im)
    for text, x, y in items:
        for ox in (-2, 2):
            for oy in (-2, 2):
                d.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0))
        d.text((x, y), text, font=font, fill=color)
    return np.asarray(im)


def draw_spine(img: np.ndarray, mask: np.ndarray, discs: list[float] | None,
               tilt_pts: tuple[tuple[float, float], tuple[float, float]] | None = None,
               scale: int = 3, deg: float | None = None,
               label_px: int = 13) -> np.ndarray:
    """Уровни дисков, контур колонны и ось; `deg` — угол решения, а не угол нарисованной линии."""
    vis = _base(img)
    cnts, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, BONE, 1)
    h, w = img.shape
    m = np.asarray(mask, bool)
    for y in discs or []:
        yy = int(round(y))
        if not 0 <= yy < h:
            continue
        xs = np.nonzero(m[yy])[0]
        # Линия по ширине колонны, а не через весь кадр.
        x0, x1 = (int(xs.min()), int(xs.max())) if len(xs) else (0, w - 1)
        pad = max(4, (x1 - x0) // 6)
        cv2.line(vis, (max(0, x0 - pad), yy), (min(w - 1, x1 + pad), yy), SPINE, 1)
    label = []
    if tilt_pts:
        (tx, ty), (bx, by) = tilt_pts
        cv2.line(vis, (int(tx), int(ty)), (int(bx), int(by)), AXIS, 2)
        shown = deg if deg is not None and deg >= 0 else abs(
            np.degrees(np.arctan((bx - tx) / max(by - ty, 1e-6))))
        label = [(f"ось {shown:.1f}°", int(bx * scale) + 8, int(by * scale) - 18)]
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return _labels(vis, label, label_px, SPINE)


def draw_spine_kp(img: np.ndarray, p: np.ndarray, deg: float | None = None,
                  art_thr: float | None = None, scale: int = 3,
                  label_px: int = 13) -> np.ndarray:
    """Центры тел, ось, крылья таза, коридор колонны и предметы — теми же функциями, что признаки.

    Рамки вокруг предметов — только если сработало правило артефактов, иначе картинка разойдётся с вердиктом.
    """
    from dxa_feat.kp_measure import GAP, HALF_W, artifact_mask, interp_x, wing_top

    h, w = img.shape
    vis = _base(img)
    pitch = float(np.median(np.diff(p[:, 1])))
    _, top = wing_top(np.asarray(img, float), p, pitch)
    cx = interp_x(p, np.arange(h, dtype=float))
    y0, y1 = int(p[0, 1]), int(top) if top is not None else h - 1
    for s in (-1, 1):
        pts = np.stack([cx[y0:y1] + s * GAP * HALF_W, np.arange(y0, y1)], 1)
        pts = np.ascontiguousarray(pts[::6].astype(np.int32)).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, CORRIDOR, 1, cv2.LINE_AA)
    art = artifact_mask(np.asarray(img, float), p, top).astype(np.uint8)
    num, lab, st, _ = cv2.connectedComponentsWithStats(art, 8)
    # одиночные пиксели по краю кадра — шум: в счёт правила идут, но не рисуются
    big = [i for i in range(1, num) if st[i, 4] >= 6]
    for i in big:
        vis[lab == i] = ART
    hit = art_thr is not None and art.sum() / art.size >= art_thr
    if hit:
        for i in big:
            x, y, bw, bh = st[i, :4]
            cv2.rectangle(vis, (x - 2, y - 2), (x + bw + 2, y + bh + 2), ART, 1)
    labels = []
    if top is not None:
        cv2.line(vis, (0, int(top)), (w - 1, int(top)), WING, 1)
        labels.append(("верх крыльев таза", 8, int(top * scale) + 6, WING))
    else:
        # Сверху: внизу стоит подпись оси.
        labels.append(("крыльев таза в кадре не видно", 8, 8, WING))
    above = p[p[:, 1] < top] if top is not None and (p[:, 1] < top).sum() >= 2 else p
    cv2.line(vis, tuple(map(int, above[0])), tuple(map(int, above[-1])), AXIS, 2, cv2.LINE_AA)
    for x, y in p:
        cv2.circle(vis, (int(x), int(y)), 3, SPINE, -1, cv2.LINE_AA)
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    if deg is not None and deg >= 0:
        bx, by = above[-1]
        labels.append((f"ось {deg:.1f}°", int(bx * scale) + 12, int(by * scale) - 14, SPINE))
    if hit and big:
        i = min(big, key=lambda j: st[j, 0])
        text = "посторонний предмет"
        # подпись не должна уезжать за правый край кадра
        lx = min(int(st[i, 0] * scale), w * scale - int(0.62 * label_px * len(text)) - 4)
        labels.append((text, max(0, lx), max(0, int(st[i, 1] * scale) - label_px - 8), ART))
    for text, lx, ly, col in labels:
        vis = _labels(vis, [(text, lx, ly)], label_px, col)
    return vis


def draw_femur(img: np.ndarray, mask: np.ndarray, pts: dict | None,
               scale: int = 3, label_px: int = 13, lt=None, line=None) -> np.ndarray:
    """Опорные точки бедра, ось диафиза, малый вертел `lt` и контур `line`, от которого меряется его выступ."""
    vis = _base(img)
    cnts, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, BONE, 1)
    names = {"prox": "медиальная опора", "isthmus": "перетяжка",
             "gt": "большой вертел", "shaft": "диафиз"}
    if pts:
        if "prox" in pts and "shaft" in pts:
            p, s = pts["prox"], pts["shaft"]
            cv2.line(vis, (int(p[0]), int(p[1])), (int(s[0]), int(s[1])), AXIS, 1)
        for key, (x, y) in pts.items():
            if key not in names:
                continue
            cv2.circle(vis, (int(x), int(y)), max(3, label_px // 9), LM, -1)
    h, w = img.shape
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    if line is not None:
        k, c, y0, y1 = line
        cv2.line(vis, (int((k * y0 + c) * scale), y0 * scale),
                 (int((k * y1 + c) * scale), y1 * scale), MEDIAL, 2, cv2.LINE_AA)
    labels = [(names[k], int(x * scale) + 7, int(y * scale) - 8)
              for k, (x, y) in (pts or {}).items() if k in names]
    vis = _labels(vis, labels, label_px)
    if lt is not None:
        vis = np.array(vis)
        x, y = int(lt[0] * scale), int(lt[1] * scale)
        cv2.circle(vis, (x, y), max(6, label_px // 3), MEDIAL, -1, cv2.LINE_AA)
        text = "малый вертел"
        lx = min(x + 10, w * scale - int(0.62 * label_px * len(text)) - 4)
        vis = _labels(vis, [(text, max(0, lx), y + 6)], label_px, MEDIAL)
    return vis



def draw_case(img: np.ndarray, region: str, side: str = "",
              mask: np.ndarray | None = None, scale: int = 3,
              tilt_deg: float | None = None, label_px: int = 13) -> np.ndarray:
    """Кадр с наложенной разметкой — одна сборка на сервис и демо-страницу."""
    if region == "spine":
        from dxa_seg.kpdet import centers
        p = centers(img)
        if p is not None and len(p) >= 2:
            from dxa_service.model import ARTIFACT_RULE
            return draw_spine_kp(img, p, deg=tilt_deg,
                                 art_thr=ARTIFACT_RULE[1] if ARTIFACT_RULE else None,
                                 scale=scale, label_px=label_px)
        from dxa_feat import spine_discs
        from dxa_feat.anatomy import _body_centers, _largest_component
        from dxa_seg.region import bone_mask

        m = bone_mask(img) if mask is None else np.asarray(mask, bool)
        col = _largest_component(m)
        cx, _ = _body_centers(img, col)
        rows = np.nonzero(~np.isnan(cx))[0]
        tp = None
        # Ось по крайним телам, как в методике и в признаке.
        if len(rows) >= 40:
            k = max(3, int(0.12 * len(rows)))
            tp = ((float(np.nanmean(cx[rows[:k]])), float(rows[:k].mean())),
                  (float(np.nanmean(cx[rows[-k:]])), float(rows[-k:].mean())))
        return draw_spine(img, col, spine_discs(img, m).get("_discs"), tp, scale,
                          deg=tilt_deg, label_px=label_px)

    from dxa_feat import femur_landmarks
    from dxa_service.femur_mask import femur_mask

    m = femur_mask(img, side) if mask is None else np.asarray(mask, bool)
    # Анатомия бедра написана под правую сторону.
    a = img[:, ::-1].copy() if side == "left" else img
    mm = m[:, ::-1].copy() if side == "left" else m
    from dxa_feat.f_hipmarks import POINTS, _predict, medial_line

    P = _predict(a)
    lt = P.get(POINTS[0]) if P else None
    return draw_femur(a, mm, femur_landmarks(mm, "right").get("_pts"), scale,
                      label_px, lt=lt, line=medial_line(mm) if lt else None)


def png_data_uri(arr: np.ndarray, scale: int = 1) -> str:
    """Картинка в PNG data-URI для ответа API."""
    import base64
    import io

    from PIL import Image

    a = arr if arr.dtype == np.uint8 else (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if scale != 1:
        a = cv2.resize(a, (a.shape[1] * scale, a.shape[0] * scale),
                       interpolation=cv2.INTER_NEAREST)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
