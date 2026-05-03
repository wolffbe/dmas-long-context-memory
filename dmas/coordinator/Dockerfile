FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY coordinator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY coordinator/main.py .
COPY coordinator/app/ ./app/
COPY shared/ ./shared/

EXPOSE 8001

CMD ["python", "main.py"]
