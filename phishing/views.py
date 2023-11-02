from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
    return render(request, 'home.html')

# def test(request):
#     return render(request, 'index.html')

def number_search(request):
    return render(request, 'number_search.html')

def text(request):
    return render(request, 'text.html')