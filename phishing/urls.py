from django.urls import path
from . import views

app_name = 'phishing'

urlpatterns = [
    path('', views.home, name='home'),
    # path('')
    path('number_search/', views.number_search, name='nuber_search'),
    path('text/', views.text, name='text')
]
