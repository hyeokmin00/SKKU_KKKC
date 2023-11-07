from django.shortcuts import render
from django.http import HttpResponse

from .models import Organizations
from django.core.paginator import Paginator


# Create your views here.
def home(request):
    return render(request, "home.html")


# 신고하기
def notify(request):
    return render(request, "notify.html")


# 탐지 결과
def result(request):
    return render(request, "result.html")


# 기관 정보
def rel_org(request):  # 기관 전체
    page = request.GET.get("page", "1")
    orgs_list = Organizations.objects.all()
    paginator = Paginator(orgs_list, 8)  # 페이지당 8개씩 보여주기
    page_obj = paginator.get_page(page)
    return render(request, "rel_organization.html", {"orgs_list": orgs_list})


def financial(request):  # 금융 기관
    page = request.GET.get("page", "1")
    orgs_list = Organizations.objects.all()
    paginator = Paginator(orgs_list, 8)  # 페이지당 8개씩 보여주기
    page_obj = paginator.get_page(page)
    context = {"question_list": page_obj}
    return render(request, "financial.html", {"orgs_list": orgs_list})


def investigative(request):  # 수사 및 신고기관
    orgs_list = Organizations.objects.all()
    return render(request, "investigative.html", {"orgs_list": orgs_list})


# 대응방법 안내
def victim_guide(request):
    return render(request, "victim_guide.html")


# ---------------------------------------------------------
# 실시간 탐지 페이지 테스트
def real_time_detectoin(request):
    return render(request, "real_time_detection.html")


# 번호 조회 페이지
def number_search(request):
    return render(request, "number_search.html")


# 실시간 탐지 시 정보 제공 동의 여부 확인
def agreement(request):
    return render(request, 'agreement.html')


def guide_test(request):
    return render(
        request,
        "guide_test.html",
    )


def test(request):
    return render(request, "home_try.html")
