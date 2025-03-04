from django.urls import path
from . import views

app_name = 'shortener'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('success/<int:pk>/', views.URLSuccessView.as_view(), name='success'),
    path('list/', views.URLListView.as_view(), name='url_list'),
    path('api/create/', views.api_create_short_url, name='api_create'),
    path('<str:short_code>/', views.redirect_to_original, name='redirect'),
]
