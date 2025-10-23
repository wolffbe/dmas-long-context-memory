FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY locomo.py .
EXPOSE 8000
CMD ["uvicorn", "locomo:app", "--host", "0.0.0.0", "--port", "8000"]