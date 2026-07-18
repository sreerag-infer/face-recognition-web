# Base image
FROM python:3.10-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create runtime folders
RUN mkdir -p uploads embeddings registration_samples

# Render sets PORT automatically
EXPOSE 5000

# Start application
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]