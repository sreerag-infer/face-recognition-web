import os
import multiprocessing

# Port mapping
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Worker processes configuration
# Note: Spawning multiple workers duplicates the InsightFace model in RAM.
# To fit inside typical 512MB/1GB memory limits on cloud platforms,
# we use 1 worker process with multiple threads.
workers = 1
threads = 4

# Server timeout
# Extented to allow lazy loading of the ONNX models at startup/first request
timeout = 120

# Logging configurations
accesslog = "-"
errorlog = "-"
loglevel = "info"
