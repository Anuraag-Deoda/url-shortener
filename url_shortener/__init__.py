"""
url_shortener Django project initialization.
"""

# Celery is optional - only import if installed
try:
    from .celery import app as celery_app
    __all__ = ('celery_app',)
except ImportError:
    celery_app = None
    __all__ = ()
