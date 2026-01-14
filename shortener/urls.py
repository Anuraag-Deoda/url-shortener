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

    # Custom Domains
    path('domains/', views.CustomDomainListView.as_view(), name='custom_domains'),
    path('domains/add/', views.CustomDomainCreateView.as_view(), name='custom_domain_add'),
    path('domains/<int:pk>/verify/', views.verify_custom_domain, name='verify_domain'),

    # QR Code
    path('qr/<str:short_code>/', views.generate_qr_code, name='qr_code'),

    # Redirect (must be last - catches all short codes)
    path('<str:short_code>/', views.redirect_to_original, name='redirect'),
]
