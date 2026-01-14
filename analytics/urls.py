from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('url/<str:short_code>/', views.URLStatsView.as_view(), name='url_stats'),
    path('api/url_data/<str:short_code>/', views.url_data_api, name='url_data_api'),
    path('api/insights/<str:short_code>/', views.ai_insights_api, name='ai_insights_api'),

    # Geographic data API for heatmaps
    path('api/geo/<str:short_code>/', views.geographic_data_api, name='geo_data_api'),
    path('api/geo/', views.geographic_data_api, name='geo_data_api_all'),
]
