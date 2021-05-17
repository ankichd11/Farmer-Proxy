from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.crop, name='crop')
]