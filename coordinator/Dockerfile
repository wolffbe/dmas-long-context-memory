FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY coordinator.py .

EXPOSE 8001

RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Ollama..."\n\
ollama serve &\n\
sleep 5\n\
\n\
echo "Pulling model ${MODEL}..."\n\
ollama pull ${MODEL}\n\
\n\
echo "Verifying model is available..."\n\
ollama list | grep ${MODEL} || (echo "Model ${MODEL} not found!" && exit 1)\n\
\n\
echo "Model ready. Starting coordinator..."\n\
exec python coordinator.py\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]