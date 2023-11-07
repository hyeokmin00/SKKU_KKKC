from django.urls import path
from . import views

app_name = 'phishing'

urlpatterns = [
    path('', views.home, name='home'),
    path('notify/', views.notify, name='notify'), #신고 페이지
    path('result/', views.result, name='result'), #결과 페이지
    path('rel_org/', views.rel_org, name='rel_org'), #관련 기관 리스트
    path('financial/', views.financial, name='financial'), #금융기관 리스트
    path('investigative/', views.investigative, name='investigative'), #수사 및 신고기관 리스트
    path('victim_guide/', views.victim_guide, name='victim_guide'), #대응방법 가이드

    # 테스트 페이지
    path('real_time_detection/', views.real_time_detectoin, name='real_time_detection'), #실시간 탐지 페이지
    path('number_search/', views.number_search, name='number_search'), #의심번호 입력
    
    path('guide_test/', views.guide_test, name='guide_test' ),    
]
