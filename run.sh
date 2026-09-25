#!/usr/bin/env bash
# Сборка и запуск сервиса оценки качества DXA.
#
#   ./run.sh batch <каталог|zip с DICOM> [каталог результата]   разовый прогон в файл
#   DXA_FORMAT=csv ./run.sh batch ...                          таблица в csv вместо xlsx
#   ./run.sh serve [порт]                                   поднять API
#   ./run.sh <вход> [выход]                                 то же, что batch
set -euo pipefail

IMAGE=dxa-qc:latest
# GPU по умолчанию: SAM 3 на CPU очень медленный. DXA_GPU=none — без GPU, device=0 — конкретная карта.
DXA_GPU="${DXA_GPU:-all}"
GPU_ARGS=()
if [[ "$DXA_GPU" != "none" ]]; then
    GPU_ARGS=(--gpus "$DXA_GPU")
fi
MODE="${1:-batch}"
if [[ "$MODE" != "batch" && "$MODE" != "serve" ]]; then
    set -- batch "$@"          # вызов без режима: ./run.sh <вход> <выход>
    MODE=batch
fi
shift

docker build -t "$IMAGE" .

if [[ "$MODE" == "serve" ]]; then
    PORT="${1:-8000}"
    echo "API на http://localhost:$PORT  (проверка: curl localhost:$PORT/health)"
    exec docker run --rm "${GPU_ARGS[@]}" -p "$PORT":8000 -v "$PWD/data":/data:ro "$IMAGE"
fi

IN="${1:?укажите каталог или zip-архив с DICOM}"
FMT="${DXA_FORMAT:-xlsx}"
OUT="${2:-$PWD/out}"
mkdir -p "$OUT"
docker run --rm "${GPU_ARGS[@]}" \
    -v "$(realpath "$IN")":/data/input:ro \
    -v "$(realpath "$OUT")":/data/output \
    "$IMAGE" python scripts/pipeline/run_service.py --input /data/input --output "/data/output/results.$FMT"
echo "результат: $OUT/results.$FMT, картинки разбора: $OUT/overlays.zip"
