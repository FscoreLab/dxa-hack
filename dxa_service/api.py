"""HTTP-API пакетной обработки; модели грузятся один раз при старте.

Ручки:
    GET  /health              — живость и что именно загружено
    POST /predict             — обработать каталог, отдать JSON со строками
    POST /predict/file        — обработать один DICOM, отдать одну строку
    POST /explain/file        — то же плюс картинка с разметкой и разбор проверок
    POST /predict/zip         — таблица и картинки разбора по каталогу одним архивом
    GET  /predict/xlsx        — то же, что /predict, но файлом по контракту ТЗ
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from dxa_service.model import RuleModel
from dxa_service.pipeline import process_dir, process_file

app = FastAPI(title="DXA quality control", version="1.0",
              description="Оценка качества денситометрических исследований")
_MODEL: RuleModel | None = None

# Обрабатывать можно только внутри этого каталога: иначе ручки раскрывают структуру ФС сервера.
DATA_ROOT = Path(os.environ.get("DXA_DATA_ROOT", "/data")).resolve()


def _safe_dir(raw: str) -> Path:
    """Каталог из запроса, ограниченный DATA_ROOT."""
    try:
        src = Path(raw).resolve()
    except (OSError, RuntimeError):
        raise HTTPException(400, "некорректный путь")
    if src != DATA_ROOT and DATA_ROOT not in src.parents:
        raise HTTPException(
            403, f"путь вне разрешённого каталога {DATA_ROOT}; "
                 f"смонтируйте данные внутрь него или задайте DXA_DATA_ROOT")
    if not (src.is_dir() or (src.is_file() and src.suffix.lower() == ".zip")):
        raise HTTPException(404, f"каталог или zip-архив не найден: {src}")
    return src


def model() -> RuleModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = RuleModel.load()
    return _MODEL


@app.on_event("startup")
def _warmup() -> None:
    """Прогрев: веса читаются на старте, а не на первом запросе."""
    m = model()
    # Без опорных распределений оценка годности ушла бы на другую шкалу — это подмена, а не деградация.
    if m is not None and m.heads and m.refs is None:
        raise RuntimeError(
            "нет models/quality_refs.json: непрерывная оценка годности ушла бы "
            "на запасную шкалу; соберите scripts/pipeline/build_quality_refs.py")
    from dxa_seg.kpdet import _load as load_kp
    from dxa_seg.sam3 import _load as load_sam3
    load_sam3()       # без измерителя и детектора тел запуск должен прерваться
    load_kp()
    try:
        from dxa_service.embed import _segmenter
        from dxa_service.femur_mask import _load
        _load()
        _segmenter()
    except Exception:
        pass          # сервис работает и без части весов, просто хуже


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    """Страница загрузки файлов в браузере."""
    return HTMLResponse((Path(__file__).parent / "web" / "index.html")
                        .read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    from dxa_service.embed import _pca, _segmenter
    from dxa_service.femur_mask import _load
    from dxa_seg.kpdet import _load as load_kp
    from dxa_seg.sam3 import _load as load_sam3
    sam3 = load_sam3()
    kp = load_kp() is not None
    return {
        "status": "ok" if sam3 and kp else "degraded",
        "models": {
            "типы нарушений": bool(model().heads),
            "маска бедра": bool(_load()),
            "энкодер эмбеддингов": bool(_segmenter()),
            "снижение размерности": bool(_pca()),
            "ось позвоночника SAM 3": sam3,
            "центры тел позвонков (детектор)": kp,
            "опорные распределения годности": model().refs is not None,
        },
    }


@app.post("/predict")
def predict(payload: dict = Body(..., example={"input": "/data/input"})) -> dict:
    """Пакетная обработка каталога: строка на изображение по контракту ТЗ."""
    src = _safe_dir(str(payload.get("input", "/data/input")))
    t0 = time.perf_counter()
    rows = process_dir(src, model())
    return {"rows": rows, "count": len(rows),
            "seconds": round(time.perf_counter() - t0, 3)}


@app.post("/predict/file")
async def predict_file(file: UploadFile) -> dict:
    """Обработать один DICOM-файл, переданный в теле запроса; вернуть строку отчёта."""
    import tempfile

    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=True) as fh:
        fh.write(data)
        fh.flush()
        row = process_file(Path(fh.name), model())
    from dataclasses import asdict
    out = asdict(row)
    # Служебное поле в контракт ответа не входит.
    out.pop("region_key", None)
    out["path_to_study"] = file.filename or ""
    return out


@app.post("/explain/file")
async def explain_file(file: UploadFile) -> dict:
    """Обработать один DICOM и вернуть вердикт, картинку с разметкой и разбор проверок."""
    # Отдельная ручка: контракт /predict/file не раздуваем картинкой. Вердикт — из analyse_file, как везде.
    import tempfile
    from dataclasses import asdict

    from dxa_service.explain import checks, tilt_of
    from dxa_service.overlay import draw_case, png_data_uri
    from dxa_service.pipeline import analyse_file

    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=True) as fh:
        fh.write(data)
        fh.flush()
        row, frame = analyse_file(Path(fh.name), model(), with_frame=True)
    out = asdict(row)
    out.pop("region_key", None)
    out["path_to_study"] = file.filename or ""
    if frame is None:
        return {**out, "overlay": None, "checks": []}
    tilt = tilt_of(model(), frame["feats"]) if frame["region"] == "spine" else None
    marks = draw_case(frame["img"], frame["region"], frame["side"], frame["mask"], label_px=26,
                      tilt_deg=tilt)
    return {**out,
            "overlay": png_data_uri(marks),
            "plain": png_data_uri(frame["img"], scale=3),
            "checks": _agree(checks(model(), frame["region"], frame["feats"]),
                             out.get("violation_type") or "")}


# проверка модели → текст нарушения; по нему проверка сверяется с вердиктом
_CHECK_VIOLATION = {"укладка позвоночника": "Некорректная укладка",
                    "укладка бедра": "Некорректная укладка",
                    "область интереса бедра": "Некорректная область интереса"}


def _agree(items: list[dict], violation_type: str) -> list[dict]:
    """Проверки моделей показываются так, как решила строка отчёта.

    На бедре типы согласуются с головой годности (`reconcile_hip`): тип может
    быть назначен ниже своего порога или снят выше него — это помечается.
    """
    got = set(filter(None, violation_type.split(";")))
    for c in items:
        name = _CHECK_VIOLATION.get(c["task"])
        if c.get("kind") != "head" or not name:
            continue
        fired = name in got
        if fired and not c["fired"]:
            c["note"] = "тип выбран по общей модели годности бедра: ближе всех к своему порогу"
        elif c["fired"] and not fired:
            c["note"] = "выше порога, но общая модель годности бедра считает снимок годным"
        c["fired"] = fired
    return items


@app.post("/predict/xlsx")
def predict_xlsx_post(payload: dict = Body(..., example={"input": "/data/input"})):
    """Обработать каталог и вернуть таблицу XLSX; путь в теле запроса (удобно для кириллицы)."""
    return _xlsx(str(payload.get("input", "/data/input")))


@app.get("/predict/xlsx")
def predict_xlsx(input: str = "/data/input") -> StreamingResponse:
    """Обработать каталог и вернуть таблицу XLSX; кириллицу в пути кодируйте или используйте POST."""
    return _xlsx(input)


@app.post("/predict/zip")
def predict_zip(payload: dict = Body(..., example={"input": "/data/input"})):
    """Таблица и картинки разбора одним архивом: results.xlsx + overlays/*.png (ТЗ п. 2.7)."""
    import zipfile

    import pandas as pd

    src = _safe_dir(str(payload.get("input", "/data/input")))
    pngs: dict[str, bytes] = {}
    df = pd.DataFrame(process_dir(src, model(), overlays=pngs))
    table = io.BytesIO()
    df.to_excel(table, index=False)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("results.xlsx", table.getvalue())
        for name, data in pngs.items():
            z.writestr(f"overlays/{name}", data)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": 'attachment; filename="results.zip"'})


def _xlsx(raw: str) -> StreamingResponse:
    import pandas as pd

    src = _safe_dir(raw)
    df = pd.DataFrame(process_dir(src, model()))
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="results.xlsx"'})
