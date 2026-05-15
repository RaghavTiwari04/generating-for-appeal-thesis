FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg, Pillow, Tesseract, OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc g++ \
    tesseract-ocr \
    libgl1 libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
