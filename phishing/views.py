from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
    # return HttpResponse("Hello!")
    return render(request, 'home.html')

# def index(request):
    # return render(request, 'phishing/home.html')

def test(request):
    return render(request, 'index.html')