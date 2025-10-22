FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0

COPY locomo.py .

EXPOSE 8000

CMD ["uvicorn", "locomo:app", "--host", "0.0.0.0", "--port", "8000"]