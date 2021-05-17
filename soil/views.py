from django.shortcuts import render

# Create your views here
from django.shortcuts import render,redirect

def soil(request):
    return render(request,'soil.html')

