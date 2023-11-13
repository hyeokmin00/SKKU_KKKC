from django.shortcuts import render
from django.http import HttpResponse

from .models import Organizations
from django.core.paginator import Paginator

from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from .models import Sentence, Text_mail
from .utils import load_kcbert_model, calculate_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



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
    # page = request.GET.get('page', '1')
    orgs_list = Organizations.objects.all()
    # paginator = Paginator(orgs_list, 8) #페이지당 8개씩 보여주기
    # page_obj = paginator.get_page(page)
    return render(request, 'rel_organization.html', {"orgs_list":orgs_list})

def financial(request): #금융 기관
    # page = request.GET.get('page', '1')
    orgs_list = Organizations.objects.all()
    # paginator = Paginator(orgs_list, 8) #페이지당 8개씩 보여주기
    # page_obj = paginator.get_page(page)
    # context = {'question_list':page_obj}
    return render(request, 'financial.html', {'orgs_list':orgs_list})

def investigative(request): #수사 및 신고기관
    orgs_list = Organizations.objects.all()
    return render(request, 'investigative.html', {'orgs_list':orgs_list})

# 대응방법 안내
def victim_guide(request):
    return render(request, 'victim_guide.html')

# ---------------------------------------------------------
# 실시간 탐지 페이지 테스트
def real_time_detectoin(request):
    return render(request, 'real_time_detection.html')

# 번호 조회 페이지
def number_search(request):
    
    return render(request, 'number_search.html')

# 실시간 탐지 시 정보 제공 동의 여부 확인
def agreement(request):
    return render(request, 'agreement.html')



################

from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from .models import Text_mail
from .utils import load_kcbert_model, calculate_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from .models import Text_mail
from .utils import load_kcbert_model, calculate_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.preprocessing import normalize

class SimilarityView(View):
    template_name = 'model_test.html'

    def get(self, request):
        return render(request, 'model_test.html')

    def post(self, request):
        input_sentence = request.POST.get('input_sentence')

        # Load KcBERT model
        kcbert_model, kcbert_tokenizer = load_kcbert_model()

        # Calculate embeddings
        input_embedding = calculate_embedding(input_sentence, kcbert_model, kcbert_tokenizer)
        input_embedding_normalized = normalize(input_embedding.reshape(1, -1))  # L2 정규화 적용

        # Calculate similarity
        all_text_mails = Text_mail.objects.all()
        similarity_scores = []

        # Set a similarity threshold
        similarity_threshold = 0.5

        for idx, text_mail in enumerate(all_text_mails):
            transcript = text_mail.transcript
            if transcript:
                db_embedding = calculate_embedding(transcript, kcbert_model, kcbert_tokenizer)
                # db_embedding = np.fromstring(transcript, dtype=float, sep=',')
                # print(db_embedding)
                # print(idx)
                if db_embedding.size > 0:
                    db_embedding_normalized = normalize(db_embedding.reshape(1, -1))  # L2 정규화 적용
                    similarity_score = cosine_similarity(input_embedding_normalized, db_embedding_normalized)[0][0]
                    # print(f"Similarity with '{transcript}': {similarity_score}")
                    if similarity_score > similarity_threshold:
                        similarity_scores.append({'text': transcript, 'similarity': similarity_score})

        print("Final Similarities:", similarity_scores)
        return JsonResponse({'input_sentence': input_sentence, 'similarities': similarity_scores})


# def guide_test(request):
#     return render(request, 'guide_test.html',)

# def test(request):
#     return render(request, 'home_try.html')
