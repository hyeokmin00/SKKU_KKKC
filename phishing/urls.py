from django.urls import path
from . import views

app_name = 'phishing'

urlpatterns = [
    path('', views.home, name='home'),
    path('test/', views.test, name='test'),
    # path('')
    path('number/', views.number, name='nuber')
]
