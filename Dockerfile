# Base Python Image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# Set work directory
WORKDIR /app

# Install system dependencies
# gcc, build-essential, cmake, and python3-dev are needed to build libraries if wheels are unavailable.
# libgl1-mesa-glx and libglib2.0-0 are critical runtime libraries for OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary persistent directories for local models/data
RUN mkdir -p uploads embeddings registration_samples

# Expose server port
EXPOSE 5000

# Start app using Gunicorn
CMD gunicorn -c gunicorn.conf.py app:app
