from transformers import BertTokenizer, BertConfig
from train import KoBERTClassifier

def get_softmax_vector(text, model_dir='./models/kobert_emotion'):
    # tokenizer는 학습 때와 같은 pretrained 모델에서 불러오기
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")  
    config = BertConfig.from_pretrained(model_dir)
    model = KoBERTClassifier.from_pretrained(model_dir, config=config)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs["logits"]
    return probs.numpy()[0]