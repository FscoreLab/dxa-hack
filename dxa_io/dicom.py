"""Чтение DXA-DICOM и приведение к единому представлению.

GE Lunar Prodigy Advance, 8 бит, PixelSpacing в выгрузке нет — см. pixel_scale.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pydicom

EXPOSED_AREA = (0x0040, 0x0303)

# Типичные анатомические размеры, мм: запасная калибровка к сантиметрам.
FEMORAL_HEAD_DIAMETER_MM = 47.0
VERTEBRAL_BODY_HEIGHT_MM = 30.0


@dataclass
class DxaImage:
    path: str
    sop_uid: str
    study_uid_tag: str
    series_uid: str
    rows: int
    cols: int
    bits_stored: int
    photometric: str
    manufacturer: str
    model: str
    software: str
    series_description: str
    exposed_area: tuple[int, int] | None
    mm_per_px_x: float | None
    mm_per_px_y: float | None
    scale_source: str
    pixel_md5: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["exposed_area"] = None if self.exposed_area is None else list(self.exposed_area)
        return d


def _tag_str(ds, name: str, default: str = "") -> str:
    v = ds.get(name, default)
    return "" if v is None else str(v)


# Паспортный размер пикселя аппарата: тегов размера в выгрузке нет.
DEVICE_MM_PER_PX_X = 0.6
DEVICE_MM_PER_PX_Y = 1.05


def pixel_scale(ds) -> tuple[float | None, float | None, str]:
    """Масштаб в мм на пиксель: PixelSpacing, иначе паспорт аппарата.

    ExposedArea (0040,0303) не годится: это накопленная облучённая площадь, а не поле зрения.
    """
    for name in ("PixelSpacing", "ImagerPixelSpacing"):
        v = ds.get(name)
        if v:
            try:
                return float(v[0]), float(v[1]), name
            except (TypeError, ValueError, IndexError):
                pass

    return DEVICE_MM_PER_PX_X, DEVICE_MM_PER_PX_Y, "device_spec"


def _looks_inverted(a: np.ndarray) -> bool:
    """Полярность по картинке: мода гистограммы приходится на обширный фон."""
    h, _ = np.histogram(a, bins=64)
    mode_bin = int(np.argmax(h))
    return mode_bin > 32          # фон в светлой половине — значит инверсия


def normalize_pixels(ds) -> np.ndarray:
    """Пиксели в float32 [0,1], кость светлая: полярность по тегу и по гистограмме.

    Диапазон по перцентилям: одиночный выброс (металл) иначе сжимает картинку.
    """
    a = ds.pixel_array.astype(np.float32)
    if _tag_str(ds, "PhotometricInterpretation").upper() == "MONOCHROME1":
        a = a.max() - a
    if _looks_inverted(a):
        a = a.max() - a
    lo, hi = (float(v) for v in np.percentile(a, [0.5, 99.5]))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    # float32 обязателен: percentile даёт float64, а torch ждёт тип весов.
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_isotropic(img: np.ndarray) -> np.ndarray:
    """Приведение к квадратному пикселю (0,6 × 1,05 мм); включается DXA_ISO=1."""
    import os

    if not os.environ.get("DXA_ISO"):
        return img
    import cv2

    h, w = img.shape
    return cv2.resize(img, (int(round(w * DEVICE_MM_PER_PX_X / DEVICE_MM_PER_PX_Y)), h),
                      interpolation=cv2.INTER_LINEAR)


def read(path: str | Path) -> tuple[DxaImage, np.ndarray]:
    ds = pydicom.dcmread(str(path))
    img = normalize_pixels(ds)
    img = to_isotropic(img)
    sx, sy, src = pixel_scale(ds)
    ea = ds.get(EXPOSED_AREA)
    ea_val = None
    if ea is not None:
        try:
            ea_val = tuple(int(x) for x in ea.value)
        except (TypeError, ValueError):
            ea_val = None

    meta = DxaImage(
        path=str(path),
        sop_uid=_tag_str(ds, "SOPInstanceUID"),
        study_uid_tag=_tag_str(ds, "StudyInstanceUID"),
        series_uid=_tag_str(ds, "SeriesInstanceUID"),
        rows=int(ds.get("Rows", 0)),
        cols=int(ds.get("Columns", 0)),
        bits_stored=int(ds.get("BitsStored", 0)),
        photometric=_tag_str(ds, "PhotometricInterpretation"),
        manufacturer=_tag_str(ds, "Manufacturer"),
        model=_tag_str(ds, "ManufacturerModelName"),
        software=_tag_str(ds, "SoftwareVersions"),
        series_description=_tag_str(ds, "SeriesDescription"),
        exposed_area=ea_val,
        mm_per_px_x=sx,
        mm_per_px_y=sy,
        scale_source=src,
        pixel_md5=hashlib.md5(ds.pixel_array.tobytes()).hexdigest(),
    )
    return meta, img


def exposures(ds) -> int:
    """Число экспозиций к моменту экспорта — объясняет наличие копий кадра."""
    v = ds.get((0x0040, 0x0301))
    return int(v.value) if v is not None else 0
