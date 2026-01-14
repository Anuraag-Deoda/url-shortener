"""
ASGI config for url_shortener project.
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django_plotly_dash.routing import get_channel_routes

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'url_shortener.settings')

# Initialize Django ASGI application early
django_asgi_app = get_asgi_application()

# Import after Django setup
from shortener.routing import websocket_urlpatterns

# Combine plotly dash routes with our custom routes
all_routes = websocket_urlpatterns + get_channel_routes()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(all_routes)
    ),
})
