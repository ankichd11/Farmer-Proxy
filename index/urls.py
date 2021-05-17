from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('crop/', include('crop.urls')),
    path('disease/', include('disease.urls')),
    path('yieldf/', include('yieldf.urls'))
]