# Dockerfile
FROM python:3.11-slim

# System deps (optional, but tzdata helps ZoneInfo)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy code + deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Default: launch via role
CMD ["python", "-m", "app.launch"]
