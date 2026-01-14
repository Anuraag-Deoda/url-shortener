from django.urls import path
from . import views

app_name = 'shortener'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('success/<int:pk>/', views.URLSuccessView.as_view(), name='success'),
    path('list/', views.URLListView.as_view(), name='url_list'),
    path('api/create/', views.api_create_short_url, name='api_create'),

    # Device Targeting
    path('url/<int:pk>/device-targets/', views.URLDeviceTargetView.as_view(), name='device_targets'),

    # Link Rotation
    path('url/<int:pk>/rotation/', views.RotationManagementView.as_view(), name='rotation'),

    # Time-Based Redirects
    path('url/<int:pk>/time-based/', views.TimeBasedRedirectView.as_view(), name='time_based'),

    # Link Settings (expiration, password, captcha)
    path('url/<int:pk>/settings/', views.URLSettingsView.as_view(), name='link_settings'),

    # Custom Domains
    path('domains/', views.CustomDomainListView.as_view(), name='custom_domains'),
    path('domains/add/', views.CustomDomainCreateView.as_view(), name='custom_domain_add'),
    path('domains/<int:pk>/verify/', views.verify_custom_domain, name='verify_domain'),

    # QR Code
    path('qr/<str:short_code>/', views.generate_qr_code, name='qr_code'),

    # Funnel Analytics
    path('funnels/', views.FunnelListView.as_view(), name='funnel_list'),
    path('funnels/create/', views.FunnelCreateView.as_view(), name='funnel_create'),
    path('funnels/<int:pk>/', views.FunnelDetailView.as_view(), name='funnel_detail'),
    path('funnels/<int:pk>/edit/', views.FunnelEditView.as_view(), name='funnel_edit'),

    # Real-Time Analytics
    path('realtime/', views.realtime_dashboard, name='realtime_dashboard'),
    path('api/realtime/stats/', views.api_realtime_stats, name='api_realtime_stats'),
    path('api/realtime/clicks/', views.api_realtime_clicks, name='api_realtime_clicks'),
    path('api/realtime/per-minute/', views.api_clicks_per_minute, name='api_clicks_per_minute'),

    # Gate (password/captcha)
    path('gate/<str:short_code>/', views.gate_view, name='gate'),

    # Redirect (must be last - catches all short codes)
    path('<str:short_code>/', views.redirect_to_original, name='redirect'),
]
