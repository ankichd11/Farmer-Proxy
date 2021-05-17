from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

def crop(request):
    return render(request, 'crop.html')