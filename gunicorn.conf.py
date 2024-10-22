"""gunicorn WSGI server configuration."""

from multiprocessing import cpu_count

bind = "unix:./gunicorn.sock"
workers = cpu_count() * 2 + 1
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"

loglevel = "info"
capture_output = True

# max_requests = 1000
# worker_class = "gevent"
