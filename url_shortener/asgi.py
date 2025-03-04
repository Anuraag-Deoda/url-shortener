"""
ASGI config for url_shortener project.
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django_plotly_dash.routing import get_channel_routes

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'url_shortener.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            get_channel_routes()
        )
    ),
})
