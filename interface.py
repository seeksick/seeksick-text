from transformers import BertTokenizer, BertForSequenceClassification
import torch

def get_softmax_vector(text, model_dir='./models'):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        softmax_probs = torch.softmax(logits, dim=1)  # (batch, 5)
    return softmax_probs.numpy()[0]  # 5차원 numpy 배열 반환

if __name__ == "__main__":
    text = input("텍스트 입력: ")
    vec = get_softmax_vector(text)
    print("softmax 확률 벡터:", vec)