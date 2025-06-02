import os

os.makedirs('logs', exist_ok=True)

workers = int(os.environ.get('GUNICORN_PROCESSES', '1'))

threads = int(os.environ.get('GUNICORN_THREADS', '1'))

timeout = int(os.environ.get('GUNICORN_TIMEOUT', '600'))

bind = os.environ.get('GUNICORN_BIND', '127.0.0.1:8000')

# Increase max request size (in bytes)
# 100MB = 100 * 1024 * 1024 = 104857600 bytes
max_request_size = 104857600

# accesslog = "./logs/access.log"
# errorlog = "./logs/error.log"
loglevel = "info"

forwarded_allow_ips = '*'

secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }