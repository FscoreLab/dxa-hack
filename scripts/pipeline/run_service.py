"""CLI пакетной обработки: каталог с DICOM → xlsx/csv по контракту ТЗ."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from dxa_service import process_dir, RuleModel

ap = argparse.ArgumentParser(description="Оценка качества DXA-исследований")
ap.add_argument("--input", "-i", required=True, help="каталог с DICOM (обходится рекурсивно)")
ap.add_argument("--output", "-o", default="results.xlsx", help="итоговый .xlsx или .csv")
ap.add_argument("--overlays", default=None,
                help="zip с картинками разбора (по умолчанию overlays.zip рядом с таблицей)")
ap.add_argument("--no-overlays", action="store_true", help="не собирать картинки разбора")
a = ap.parse_args()

from dxa_seg.sam3 import _load as load_sam3
load_sam3()       # обязательный измеритель проверяем до обработки каталога

# Без энкодера колонки эмбеддинга молча становятся -1 и вердикты меняются.
from dxa_service.embed import _segmenter
if not _segmenter():
    print("ВНИМАНИЕ: энкодер эмбеддингов не поднялся — вердикты будут отличаться "
          "от тех, на которых измерено качество. Проверьте HF_HOME и веса SAM.",
          file=sys.stderr)

t0 = time.perf_counter()
pngs = None if a.no_overlays else {}
rows = process_dir(a.input, RuleModel.load(), overlays=pngs)
df = pd.DataFrame(rows)
out = Path(a.output)
(df.to_excel(out, index=False) if out.suffix == ".xlsx" else df.to_csv(out, index=False))
if pngs is not None:
    import zipfile
    zpath = Path(a.overlays) if a.overlays else out.with_name("overlays.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for name, data in pngs.items():
            z.writestr(name, data)
    print(f"картинок разбора: {len(pngs)} -> {zpath}")

ok = (df.processing_status == "Success").sum()
print(f"обработано {len(df)} изображений за {time.perf_counter()-t0:.1f} с")
print(f"успешно {ok}, ошибок {len(df)-ok}, среднее время на файл "
      f"{df.time_of_processing.mean():.3f} с")
if ok:
    d = df[df.processing_status == "Success"]
    print(f"с нарушением: {int((d.quality_class==1).sum())} из {ok}")
    print(d.anatomical_region.value_counts().to_string())
print(f"записано: {out}")
