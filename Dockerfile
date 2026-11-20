FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hairclip_service.py api.py setup_models.py ./

ENV HAIRCLIP_BASE_DIR=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
