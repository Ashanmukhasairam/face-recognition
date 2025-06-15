import os

# Server socket
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes
workers = 1  # For face recognition, we'll use a single worker
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 2

# Process naming
proc_name = 'face-recognition'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# SSL
keyfile = None
certfile = None

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Deployment environment
raw_env = [
    f"PRODUCTION=true",
    f"MONGODB_URI={os.getenv('MONGODB_URI', '')}",
    f"DB_NAME={os.getenv('DB_NAME', 'face_recognition')}"
]
