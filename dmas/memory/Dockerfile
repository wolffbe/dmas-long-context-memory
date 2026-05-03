FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY memory/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY memory/main.py .
COPY memory/app/ ./app/
COPY shared/ ./shared/

EXPOSE 8002

CMD ["python", "main.py"]
