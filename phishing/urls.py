from django.urls import path
from . import views

app_name = 'phishing'

urlpatterns = [
    path('', views.home, name='home'),
    path('notify/', views.notify, name='notify'),

    # path('')
    path('number_search/', views.number_search, name='nuber_search'),
    path('mail/', views.SMS_mail, name='SMS_mail'),
    path('rel_org/', views.rel_org, name='rel_org'),
    path('call/', views.call, name='call'),
    path('text/', views.text, name='text'),
    path('capture/', views.capture, name='capture'),
    path('victim_guide/', views.victim_guide, name='victim_guide'),
]
