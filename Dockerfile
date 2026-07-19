# Base image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download InsightFace model DURING IMAGE BUILD
RUN mkdir -p /root/.insightface/models && \
    wget -O /tmp/buffalo_s.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip && \
    unzip /tmp/buffalo_s.zip -d /root/.insightface/models && \
    rm /tmp/buffalo_s.zip

# Copy project
COPY . .

# Runtime folders
RUN mkdir -p uploads embeddings registration_samples

EXPOSE 5000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]