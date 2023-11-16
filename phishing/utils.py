from transformers import AutoModel, AutoTokenizer
import torch

def load_kcbert_model():
    # KcBERT tokenizer가 있는 디렉토리 경로 지정
    tokenizer_directory = "C:\\Users\\03123\\.cache\\huggingface\\hub\\models--beomi--kcbert-base\\snapshots\\0f2f3f8ce58a3e2dab3f4c9f547cbb612061c2ed"  # 슬래시 방향 수정
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)
    model = AutoModel.from_pretrained(tokenizer_directory)
    return model, tokenizer

def calculate_embedding(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors="pt",max_length=128, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    embedding = embedding.astype(float)
    return embedding