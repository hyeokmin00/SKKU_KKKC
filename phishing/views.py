from django.shortcuts import render
from django.http import HttpResponse

from .models import Organizations
from django.core.paginator import Paginator

from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from .models import Sentence, Text_mail, Phone_numbers, Account_numbers
from .utils import load_kcbert_model, calculate_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

##### 테스트 페이지
def guide_test(request):
    return render(request, 'guide_test.html',)

# def test(request):
#     return render(request, 'home_try.html')

# Create your views here.
def home(request):
    return render(request, 'home.html')

# 신고하기
def notify(request):
    return render(request, 'notify.html')

# 번호 조회 결과 : 위험도 높음
def result_num_high(request):
    return render(request, 'result_num_high.html')

# 번호 조회 결과 : 위험도 낮음
def result_num_low(request):
    return render(request, 'result_num_low.html')

# 모델 결과 : 위험도 높음
def result_high(request):
    is_blocking_active = False
    return render(request, 'result_model_high.html', {'is_blocking_active':is_blocking_active})

# 모델 결과 : 위험도 낮음
def result_low(request):
    return render(request, 'result_model_low.html')

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

# 실시간 탐지 페이지
def real_time_detectoin(request):
    return render(request, 'real_time_detection.html')

# 정밀검사 페이지
def text_detection(request):
    return render(request, 'text_detection.html')

# 번호 조회 페이지
def number_search(request):
    search_number = '' # 기본값 설정
    
    if request.method == "POST":
        search_number = request.POST.get('search_number', '')

        # 데이터베이스에서 조회
        phone_numbers = Phone_numbers.objects.filter(phone_number=search_number)
        account_numbers = Account_numbers.objects.filter(account_number=search_number)

        # 결과에 따라 다른 템플릿으로 렌더링
        # 전화번호에 있으면
        if phone_numbers.exists():
            phone_numbers_obj = phone_numbers.first()
            phone_numbers_obj.search_cnt += 1
            phone_numbers_obj.save()
            return render(request, 'result_num_high.html', {'phone_numbers': phone_numbers, 'account_numbers': account_numbers, 'search_number': search_number})
        # 계좌번호에 있으면
        elif account_numbers.exists():
           account_numbers_obj = account_numbers.first()
           account_numbers_obj.search_cnt += 1
           account_numbers_obj.save()
           return render(request, 'result_num_high.html', {'phone_numbers': phone_numbers, 'account_numbers': account_numbers, 'search_number': search_number})
        else:
            return render(request, 'result_num_low.html', {'search_number': search_number})
        
    return render(request, 'number_search.html', {'search_number': search_number})


# 실시간 탐지 시 정보 제공 동의 여부 확인
def agreement(request):
    return render(request, 'agreement.html')



################
# from django.shortcuts import render
# from django.views import View
# from django.http import JsonResponse
# from .models import Text_mail
# from .utils import load_kcbert_model, calculate_embedding
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# from django.shortcuts import render
# from django.views import View
# from django.http import JsonResponse
# from .models import Text_mail
# from .utils import load_kcbert_model, calculate_embedding
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# from sklearn.preprocessing import normalize

#KcBERT
# class SimilarityView(View):
#     template_name = 'model_test.html'

#     def get(self, request):
#         return render(request, 'model_test.html')

#     def post(self, request):
#         input_sentence = request.POST.get('input_sentence')

#         # Load KcBERT model
#         kcbert_model, kcbert_tokenizer = load_kcbert_model()

#         # Calculate embeddings
#         input_embedding = calculate_embedding(input_sentence, kcbert_model, kcbert_tokenizer) 
#         input_embedding_normalized = normalize(input_embedding.reshape(1, -1))  #db 데이터와 비교 위해 정규화 # L2 정규화 적용

#         # Calculate similarity
#         all_text_mails = Text_mail.objects.all()
#         similarity_scores = []

#         # Set a similarity threshold
#         similarity_threshold = 0.7 #유사도 판단 기준값

#         for idx, text_mail in enumerate(all_text_mails):
#             transcript = text_mail.message
#             if transcript:
#                 db_embedding = calculate_embedding(transcript, kcbert_model, kcbert_tokenizer) #input 데이터 임베딩과 같은 방식 사용 필요
#                 # db_embedding = np.fromstring(transcript, dtype=float, sep=',')
#                 # print(db_embedding)
#                 # print(idx)
#                 if db_embedding.size > 0:
#                     db_embedding_normalized = normalize(db_embedding.reshape(1, -1))  # L2 정규화 적용
#                     similarity_score = cosine_similarity(input_embedding_normalized, db_embedding_normalized)[0][0]
#                     # print(f"Similarity with '{transcript}': {similarity_score}")
#                     if similarity_score > similarity_threshold: #유사도 0.5 초과면 유사도 있다고 판단, 리스트에 내용 추가
#                         similarity_scores.append({'text': transcript, 'similarity': similarity_score})

#         print("Final Similarities:", similarity_scores) #html로 결과 반환
#         return JsonResponse({'input_sentence': input_sentence, 'similarities': similarity_scores})

import pandas as pd
from django.shortcuts import render
import torch
import torch.nn.functional as F
# CSV 파일에서 데이터를 읽어옴
data = pd.read_csv('phishing\KcBERT_Input_Embedding.csv')
# 캐시용 딕셔너리 생성
embedding_cache = {}

class SimilarityView(View):
    template_name = 'model_test.html'

    def get(self, request):
        return render(request, 'model_test.html')

    def post(self, request):
        similarity_scores = []
        similarity_threshold = 0.7  # 유사도 판단 기준값
        # Load the precomputed embeddings from the CSV file
        for index, row in data.iterrows():
            transcript = row['Input_data']
            if transcript:
                try:
                    # 데이터가 'tensor'로 시작하는 문자열인 경우
                    if isinstance(transcript, str) and transcript.startswith("tensor("):
                        # 이 부분에서 텐서 값을 파싱하여 사용하는 코드 추가
                        tensor_value = torch.tensor(eval(transcript))
                        transcript_tensor = tensor_value.numpy()  # 텐서를 NumPy 배열로 변환
                    else:
                        # 데이터가 문자열로 표현된 리스트인 경우
                        db_embedding = ast.literal_eval(transcript)
                        transcript_tensor = np.array(db_embedding)

                    # 나머지 코드는 그대로 유지
                    if transcript in embedding_cache:
                        similarity_score = embedding_cache[transcript]
                    else:
                        db_embedding_tensor = torch.tensor(eval(db_embedding))  # 변환된 텍스트를 텐서로 변환
                        # Calculate cosine similarity between transcript and db_embedding tensors
                        similarity_score = F.cosine_similarity(transcript_tensor.unsqueeze(0), db_embedding_tensor.unsqueeze(0)).item()
                        embedding_cache[transcript] = similarity_score

                    if similarity_score > similarity_threshold:
                        similarity_scores.append({'text': transcript, 'similarity': similarity_score})

                except (ValueError, SyntaxError) as e:
                    print(f"Error processing transcript: {e}")
                    # 오류 발생 시 계속 진행하지 않고 다음 데이터로 넘어갈 수 있도록 continue 사용

        similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
        top_similarity_scores = similarity_scores[:10]
        print("Top 10 Similarities:", top_similarity_scores)
        return JsonResponse({'similarities': top_similarity_scores})

# import pandas as pd
# from django.shortcuts import render
# import torch
# import torch.nn.functional as F
# import ast
# import numpy as np
# import re
# # CSV 파일에서 데이터를 읽어옴
# data = pd.read_csv('phishing\KcBERT_Input_Embedding.csv')
# # 캐시용 딕셔너리 생성
# embedding_cache = {}

# class SimilarityView(View):
#     template_name = 'model_test.html'

#     def get(self, request):
#         return render(request, 'model_test.html')

#     def post(self, request):
#         similarity_scores = []
#         similarity_threshold = 0.7  # 유사도 판단 기준값
#         # Load the precomputed embeddings from the CSV file
#         for index, row in data.iterrows():
#             transcript = row['Input_data']
#             if transcript:
#                 try:
#                     # 데이터가 'tensor'로 시작하는 문자열인 경우
#                     if isinstance(transcript, str) and transcript.startswith("tensor("):
#                         # 정규표현식을 사용하여 'tensor([...])'에서 'tensor('와 ')'를 제외한 부분을 추출
#                         match = re.search(r'tensor\((.*)\)', transcript)
#                         # 추출한 문자열을 다시 텐서로 변환
#                         if match:
#                             tensor_str = match.group(1)
#                             # 추출한 문자열을 다시 텐서로 변환
#                             tensor_value = torch.tensor(ast.literal_eval(tensor_str))
#                             transcript_tensor = tensor_value.numpy()  # 텐서를 NumPy 배열로 변환
#                         else:
#                             continue  # match가 없으면 다음 데이터로 건너뜀
#                     else:
#                         # 데이터가 문자열로 표현된 리스트인 경우
#                         db_embedding = ast.literal_eval(transcript)
#                         transcript_tensor = np.array(db_embedding)

#                     # 나머지 코드는 그대로 유지
#                     if transcript in embedding_cache:
#                         similarity_score = embedding_cache[transcript]
#                     else:
#                         db_embedding_tensor = torch.tensor(eval(db_embedding))  # 변환된 텍스트를 텐서로 변환
#                         # Calculate cosine similarity between transcript and db_embedding tensors
#                         similarity_score = F.cosine_similarity(torch.tensor(transcript_tensor).unsqueeze(0), db_embedding_tensor.unsqueeze(0)).item()
#                         embedding_cache[transcript] = similarity_score

#                     if similarity_score > similarity_threshold:
#                         similarity_scores.append({'text': transcript, 'similarity': similarity_score})

#                 except (ValueError, SyntaxError) as e:
#                     print(f"Error processing transcript: {e}")
#                     # 오류 발생 시 계속 진행하지 않고 다음 데이터로 넘어갈 수 있도록 continue 사용

#         similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
#         top_similarity_scores = similarity_scores[:10]
#         print("Top 10 Similarities:", top_similarity_scores)
#         return JsonResponse({'similarities': top_similarity_scores})
