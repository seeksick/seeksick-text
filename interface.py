import torch
from transformers import BertTokenizer, BertConfig
from train import KoBERTClassifier

def get_softmax_vector(text, model_dir='./models/kobert_emotion'):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    config = BertConfig.from_pretrained(model_dir)
    model = KoBERTClassifier.from_pretrained(model_dir, config=config)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs["logits"]  # softmax된 확률 벡터
    return probs.numpy()[0]  # 5차원 numpy 배열 반환

if __name__ == "__main__":
    text = input("텍스트 입력: ")
    vec = get_softmax_vector(text)
    print("softmax 확률 벡터:", vec)