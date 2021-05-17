from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.disease, name='disease'),
    path('submit/', views.submit, name='submit-d')
]