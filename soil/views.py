from django.shortcuts import render

# Create your views here
from django.shortcuts import render,redirect

def home(request):
    return render(request,'index.html')

