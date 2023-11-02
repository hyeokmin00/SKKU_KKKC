from django.urls import path
from . import views

app_name = 'phishing'

urlpatterns = [
    path('', views.home, name='home'),
    # path('')
    path('number_search/', views.number_search, name='nuber_search'),
    path('mail/', views.SMS_mail, name='SMS_mail'),
    path('rel_org/', views.rel_org, name='rel_org'),
    path('text/', views.text, name='text')
]
