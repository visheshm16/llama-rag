import os

os.makedirs('logs', exist_ok=True)

workers = 1

worker_connections = 1000

timeout = int(os.environ.get('GUNICORN_TIMEOUT', '600'))

bind = os.environ.get('GUNICORN_BIND', '127.0.0.1:8000')

# Increase max request size (in bytes)
# 100MB = 100 * 1024 * 1024 = 104857600 bytes
max_request_size = 104857600

capture_output = True

accesslog = "./logs/access.log"
errorlog = "./logs/error.log"
loglevel = "info"

forwarded_allow_ips = '*'

secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }

# Disable request buffering for streaming
# This is important for real-time streaming
limit_request_line = 0
limit_request_fields = 100
limit_request_field_size = 8190