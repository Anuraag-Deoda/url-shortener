"""URL configuration for url_shortener project."""

from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', RedirectView.as_view(url='/shortener/', permanent=True)),
    path('shortener/', include('shortener.urls')),
    path('analytics/', include('analytics.urls')),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]
