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

# # 실시간 탐지 페이지
# def real_time_detectoin(request):
#     return render(request, 'real_time_detection.html')

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

# class SimilarityView(View):
#     template_name = 'model_test.html'

#     def get(self, request):
#         return render(request, 'model_test.html')

#     def post(self, request):
#         all_text_mails = Text_mail.objects.all()

#         text_embedding = Text_mail.objects.values('message_embedding')

#         # for i in text_embedding[:10]:
#         #     print(i)

#         similarity_scores = []
#         similarity_threshold = 0.7  # 유사도 판단 기준값
#         # Load the precomputed embeddings from the CSV file
#         for index, row in data.iterrows(): #csv 파일 데이터 임베딩
#             transcript = row['Input_data']
#             if transcript:
#                 try:
#                     # 데이터가 'tensor'로 시작하는 문자열인 경우
#                     if isinstance(transcript, str) and transcript.startswith("tensor("):
#                         # 이 부분에서 텐서 값을 파싱하여 사용하는 코드 추가
#                         # tensor_value = torch.tensor(eval(transcript[7:-1]))
#                         # transcript_tensor = tensor_value.numpy()  # 텐서를 NumPy 배열로 변환
#                         transcript_tensor = np.array(eval(transcript[7:-1])) #해당 열의 값을 2차원 리스트 형태로 가져와서 numpy로 변환
                        
#                     else:
#                         # 데이터가 문자열로 표현된 리스트인 경우
#                         db_embedding = ast.literal_eval(transcript[7:-1])
#                         transcript_tensor = np.array(db_embedding)

#                         # transcript_tensor = 

#                     # 나머지 코드는 그대로 유지
#                     if transcript in embedding_cache:
#                         similarity_score = embedding_cache[transcript]
#                     else:
#                         db_embedding_tensor = torch.tensor(eval(db_embedding))  # 변환된 텍스트를 텐서로 변환
#                         # db_embedding_tensor = np.array(transcript[7:-1])
#                         # Calculate cosine similarity between transcript and db_embedding tensors
#                         similarity_score = F.cosine_similarity(transcript_tensor.unsqueeze(0), db_embedding_tensor.unsqueeze(0)).item()
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


# class SimilarityView(View):
#     template_name = 'model_test.html'
#     def get(self, request):
#         return render(request, 'model_test.html')
#     def post(self, request):
#         similarity_scores = []
#         similarity_threshold = 0.7  # 유사도 판단 기준값
#         all_text_mails = Text_mail.objects.all()
#         text_embedding = Text_mail.objects.values('message_embedding')
#         # print(text_embedding[:1])

#         for i in all_text_mails[:10]:
#             # test_dict = {
#             #     'id' : str(i.id),
#             #     'message' : i.message,
#             #     'message_embedding' : i.message_embedding
#             # }
#             print(i.id)
#             print(i.phone_number)
#             # print(i.message_embedding)
#             print(i.message_embedding[7:-1])

#             transcript_tensor = np.array(eval(i.message_embedding[7:-1]))

#         for index, row in data.iterrows():
#             transcript = row['Input_data']
#             if isinstance(transcript, str) and transcript.startswith("tensor("):
#                 np_value = np.array(eval(transcript[7:-1]))
#                 # np_value = np_value.reshape((768,))

#                 for db_embedding in text_embedding:
#                     db_value = np.array(eval(db_embedding['message_embedding'][7:-1])) # db message_embedding 열 받아온거 db_value에 저장
#                     # db_value = db_value.reshape((768,))
#                     # 두 텐서의 코사인 유사도 계산
#                     similarity_score = np.dot(np_value, db_value) / (np.linalg.norm(np_value) * np.linalg.norm(db_value))
#                     if similarity_score > similarity_threshold:
#                         similarity_scores.append({'text': transcript, 'similarity': similarity_score})
#         similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
#         top_similarity_scores = similarity_scores[:10]
#         print("Top 10 Similarities:", top_similarity_scores)
#         return JsonResponse({'similarities': top_similarity_scores})

############### FileNotFoundError
# class SimilarityView(View):
#     template_name = 'model_test.html'
#     def get(self, request):
#         return render(request, 'model_test.html')
#     def post(self, request):
#         similarity_scores = []
#         similarity_threshold = 0.7  # 유사도 판단 기준값
#         # 클라이언트로부터 전송된 'Input_data'를 가져옴
#         transcript = request.POST.get('Input_data')
#         all_text_mails = Text_mail.objects.all()
#         for text_mail in all_text_mails:
#             db_value = np.array(torch.load(text_mail.message_embedding)[7:-1])
#             np_value = np.array(torch.load(transcript[7:-1]))
#             similarity_score = np.dot(np_value, db_value) / (np.linalg.norm(np_value) * np.linalg.norm(db_value))
#             if similarity_score > similarity_threshold:
#                 similarity_scores.append({'text': transcript, 'similarity': similarity_score})
#         similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
#         top_similarity_scores = similarity_scores[:10]
#         print("Top 10 Similarities:", top_similarity_scores)
#         return JsonResponse({'similarities': top_similarity_scores})
import pandas as pd
from django.shortcuts import render
import torch
import torch.nn.functional as F
import ast
import json
# CSV 파일에서 데이터를 읽어옴
data = pd.read_csv('phishing\KcBERT_Input_test.csv')
# 캐시용 딕셔너리 생성
embedding_cache = {}
class SimilarityView(View):
    template_name = 'model_test.html'
    def get(self, request):
        return render(request, 'real_time_detection.html') #model_test.html
    def post(self, request):
        similarity_scores = []
        similarity_threshold = 0.7  # 유사도 판단 기준값
        transcript = data['Input_data']
        input_text = data['Text']
        input_num = data['Phone_number']
        all_text_mails = Text_mail.objects.all()
        # text_embedding = Text_mail.objects.values('message_embedding')
        # print(text_embedding[:1])
        # print(transcript)
        # print(input_text)
        # print(input_num)
        transcript = transcript.values[0]
        input_text = input_text.values[0]
        input_num = input_num.values[0]
        # print()
        # print(transcript)
        # print(input_text)
        # print(input_num)

        # 테스트 말고 실제 운영 가정하는 경우 input data가 text로 입력 들어왔을 때를 고려해 임베딩 진행 과정 필요할 듯

        input_data = []

        for i in all_text_mails[:50]:
            # print(i.phone_number)
            db_value = np.array(ast.literal_eval(i.message_embedding[7:-1])).reshape(-1, 1)
            # print(np.array(ast.literal_eval(i.message_embedding)[7:-1]))
            np_value = np.array(eval(transcript[7:-1]))

            similarity_score = np.dot(np_value, db_value) / (np.linalg.norm(np_value) * np.linalg.norm(db_value))
            # print(similarity_score.shape)

            if similarity_score > similarity_threshold:
                similarity_scores.append({
                    'text': i.message,
                    # 'text': transcript, 
                    # 'similarity': similarity_score
                    'similarity': similarity_score[0][0] # 행렬연산시 변경되어야 함.

                    })
                # input data 중복없이 추가하기
                if input_text not in [item['text'] for item in input_data] and input_num not in [item['phone_number'] for item in input_data]:
                    input_data.append({
                        'text':input_text,
                        'phone_number':input_num
                    })
        
        # input_data = json.dumps(input_data, ensure_ascii=False)
        result_cnt = len(input_data) #threshold 넘는 input data 개수 = 의심건수
        cat = "실시간"

        similarity_scores.sort(key=lambda x: x['similarity'], reverse=True) #유사도 높은 순으로 정렬
        top_similarity_scores = similarity_scores[:10]
        # print(type(similarity_scores))
        print(len(input_data))
        print('input data :', input_data)
        print("Top 10 Similarities : ", top_similarity_scores)
        
        # input data만 보여주기
        # return JsonResponse({'inputs':input_data},  safe=False, json_dumps_params={'ensure_ascii': False})
        # 유사도 결과만 보여주기
        # return JsonResponse({'similarities': top_similarity_scores},  safe=False, json_dumps_params={'ensure_ascii': False})

        # 결과에 따라 다른 페이지로 연결
        # input data가 유사도 기준값을 넘는 경우
        if similarity_scores:
            result_data = [{'text':item['text'], 'sim':item['similarity']} for item in top_similarity_scores]
            return render(request, 'result_model_high.html', {'inputs':input_data, 'result_cnt':result_cnt, 'cat':cat, 'similarities':top_similarity_scores})
        # input data가 유사도 기준값을 넘지 않는 경우
        else:
            return render(request, 'result_model_low.html', {'inputs':input_data, 'cat':cat})
        
# 실시간 탐지 페이지
# SimilarityView를 실시간 탐지 페이지 html에 바로 연결하면 돼서 이 부분은 필요 없을 듯
# 버튼 클릭 없이 페이지를 로드했을 때 POST 요청을 보내 실행하고 싶다면 사용 고려해보기
# def real_time_detectoin(request):
#     rt_detection = SimilarityView() #SimilarityView 클래스의 인스턴스를 생성
#     return render(request, 'real_time_detection.html', {'similarity_data':rt_detection})
    # return rt_detection.post(request)
    # SimilarityView 인스턴스의 post 메서드를 호출 -> SimilarityView의 POST 요청 처리 로직이 실행
    # post 메서드가 반환한 결과는 return rt_detection.post(request)를 통해 real_time_detection 함수의 반환 값이 됨

##############################################
# class SimilarityView(View):
#     template_name = 'model_test.html'
#     def get(self, request):
#         return render(request, 'model_test.html')
#     def post(self, request):
#         similarity_scores = []
#         similarity_threshold = 0.7  # 유사도 판단 기준값
#         transcript = data['Input_data']
#         all_text_mails = Text_mail.objects.all()
#         # text_embedding = Text_mail.objects.values('message_embedding')
#         # print(text_embedding[:1])
#         transcript = transcript.values[0]


#         # dot product시 하나씩 유사도를 계산하는 것은 비효율 (1,768)*(768,1) 을 7400번 하는 것은 불필요
#         # ==> (7400, 768) * (768, 1)

#         # db_value -> 행렬로 변환해 input data와 연산하기

#         for i in all_text_mails[:20]:
#             print(i.phone_number)
#             db_value = np.array(ast.literal_eval(i.message_embedding[7:-1])).reshape(-1, 1)
#             # print(np.array(ast.literal_eval(i.message_embedding)[7:-1]))
#             np_value = np.array(eval(transcript[7:-1]))

#             similarity_score = np.dot(np_value, db_value) / (np.linalg.norm(np_value) * np.linalg.norm(db_value))
#             print(similarity_score.shape)

#             if similarity_score > similarity_threshold:
#                 similarity_scores.append({
#                     'text': i.message,
#                     # 'text': transcript, 
#                     # 'similarity': similarity_score
#                     'similarity': similarity_score[0][0] # 행렬연산시 변경되어야 함.

#                     })
#         similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
#         top_similarity_scores = similarity_scores[:10]
#         # print(type(similarity_scores))
#         print("Top 10 Similarities : ", top_similarity_scores)
#         return JsonResponse({'similarities': top_similarity_scores},  safe=False, json_dumps_params={'ensure_ascii': False})

#         # return JsonResponse({'similarities': top_similarity_scores})

##### 테스트 페이지
def test_result(request):
    return render(request, 'ppt_result.html')