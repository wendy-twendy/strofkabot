FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY strofkabot/ ./strofkabot/
COPY data/artan_quotes.yaml ./data/
COPY models/ ./models/
COPY utils/ ./utils/

# Data directory will be mounted as a volume
VOLUME /app/data

ENV PYTHONPATH=/app

CMD ["python", "strofkabot/llumi.py"]
