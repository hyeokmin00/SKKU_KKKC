from transformers import AutoModel, AutoTokenizer
import torch

def load_kcbert_model():
    model_name = "beomi/kcbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer
def calculate_embedding(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    embedding = embedding.astype(float)
    return embedding







