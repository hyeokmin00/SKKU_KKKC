from django.shortcuts import render
from django.http import HttpResponse

from .models import Organizations
from django.core.paginator import Paginator

# Create your views here.
def home(request):
    return render(request, 'home.html')

# 신고하기
def notify(request):
    return render(request, 'notify.html')

# 탐지 결과
def result(request):
    return render(request, 'result.html')

# 기관 정보
def rel_org(request): #기관 전체
    page = request.GET.get('page', '1')
    orgs_list = Organizations.objects.all()
    paginator = Paginator(orgs_list, 8) #페이지당 8개씩 보여주기
    page_obj = paginator.get_page(page)
    return render(request, 'rel_organization.html', {"orgs_list":orgs_list})

def financial(request): #금융 기관
    page = request.GET.get('page', '1')
    orgs_list = Organizations.objects.all()
    paginator = Paginator(orgs_list, 8) #페이지당 8개씩 보여주기
    page_obj = paginator.get_page(page)
    context = {'question_list':page_obj}
    return render(request, 'financial.html', {'orgs_list':orgs_list})

def investigative(request): #수사 및 신고기관
    orgs_list = Organizations.objects.all()
    return render(request, 'investigative.html', {'orgs_list':orgs_list})

# 대응방법 안내
def victim_guide(request):
    return render(request, 'victim_guide.html')

def test(request):
    return render(request, 'home_try.html')

def number_search(request):
    return render(request, 'number_search.html')

def SMS_mail(request):
    return render(request, 'SMS_mail.html')

def text(request):
    return render(request, 'text.html')

def capture(request):
    return render(request, 'capture.html')

def call(request):
    return render(request, 'call.html')
