from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('url/<str:short_code>/', views.URLStatsView.as_view(), name='url_stats'),
    path('api/url_data/<str:short_code>/', views.url_data_api, name='url_data_api'),
    path('api/insights/<str:short_code>/', views.ai_insights_api, name='ai_insights_api'),
]
