FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY responder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY responder/main.py .
COPY responder/app/ ./app/
COPY shared/ ./shared/

EXPOSE 8006

CMD ["python", "main.py"]
