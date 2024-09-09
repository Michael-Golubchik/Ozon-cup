FROM python:3.10-slim

# Для LightGBM не хватает библиотеки libgomp1, ее ставим
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Для загрузки файла в создаваемый образ с Google drive, с помощью gdown
RUN apt-get update && apt-get install -y \
   wget \
   && pip install gdown \
   && rm -rf /var/lib/apt/lists/*

WORKDIR /app
VOLUME /app/data
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3","/app/make_submission.py"]

COPY . /app

RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    chmod +x /app/entrypoint.sh /app/baseline.py /app/make_submission.py

# Скачаем веса моделей
# RUN gdown 137mwbRtlyfAqIMBSqljBqcI-Rb5wHElL -O /app/baseline.pkl
# RUN gdown 1cMVEKbT4SSRuUBtimqv4x0kDS-xZqzCs -O /app/bert.pkl
