FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY memory/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps cognee-community-vector-adapter-qdrant==0.2.4

COPY memory/main.py .
COPY memory/app/ ./app/
COPY shared/ ./shared/

EXPOSE 8002

CMD ["python", "main.py"]
