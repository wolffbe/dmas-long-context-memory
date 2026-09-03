FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY benchmark/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY benchmark/main.py .
COPY benchmark/app/ ./app/
COPY shared/ ./shared/

EXPOSE 8000

CMD ["python", "main.py"]
