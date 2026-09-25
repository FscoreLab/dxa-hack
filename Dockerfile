# Базовый образ зафиксирован по digest (ТЗ: версии среды выполнения фиксируются).
FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9

# libGL и libglib нужны opencv; ставим до зависимостей ради кеша слоёв.
# fonts-dejavu-core — кириллица подписей оверлея, без него подписи молча пропадают.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.lock ./
# requirements.lock — все пакеты с версиями, включая транзитивные (pip freeze образа).
# Таймаут и повторы: torch тяжёлый, на медленном канале установка срывается.
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.lock
RUN python -c "from transformers import Sam2Model, Sam2Processor, Sam3Model, Sam3Processor" \
    && pip check

COPY dxa_io dxa_io
COPY dxa_seg dxa_seg
COPY dxa_feat dxa_feat
COPY dxa_eval dxa_eval
COPY dxa_service dxa_service
COPY scripts scripts
COPY models models

# Веса SAM 3 кладутся в образ: в контуре заказчика сети наружу нет.
ENV HF_HOME=/app/.hfcache HUGGINGFACE_HUB_CACHE=/app/.hfcache HF_HUB_OFFLINE=1
COPY .hfcache/sam3 /app/.hfcache/sam3
ENV DXA_SAM3_PATH=/app/.hfcache/sam3 DXA_SAM3_REQUIRED=1

ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4
# По умолчанию API; разовый прогон: docker run ... dxa-qc python scripts/pipeline/run_service.py --input ...
EXPOSE 8000
CMD ["uvicorn", "dxa_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
