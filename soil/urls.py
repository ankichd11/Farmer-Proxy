from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('crop/', include('crop.urls')),
    path('disease/', include('disease.urls'))
]