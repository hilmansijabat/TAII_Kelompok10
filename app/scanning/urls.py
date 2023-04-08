from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('scanning/', views.scanning_request, name='scanning'),
    path('scanning/detail/', views.detail),
    path('scanning/process', views.scanning_process),
    path('scanning/predict', views.scanning_image)
]