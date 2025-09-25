# Dockerfile
FROM python:3.11-slim

# System deps: git for pip VCS installs, tzdata for ZoneInfo
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY . /app

# Launch (trainer/trader chosen by APP_ROLE)
CMD ["python", "-m", "app.launch"]
