import torch
from transformers import BertTokenizer, BertConfig
from kobert_emotion.model import KoBERTForEmotion

def load_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    config = BertConfig.from_pretrained(model_dir)
    model = KoBERTForEmotion.from_pretrained(model_dir, config=config)
    model.cuda()
    model.eval()
    return model, tokenizer

def predict(text, model, tokenizer):
    with torch.no_grad():
        encoding = tokenizer(
            text, truncation=True, padding='max_length', max_length=64, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].cuda()
        attention_mask = encoding['attention_mask'].cuda()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = outputs['logits'].cpu().numpy()[0]
        return probs  # [행복, 우울, 놀람, 분노, 중립] 순서

if __name__ == "__main__":
    model, tokenizer = load_model('kobert_emotion_trained')
    text = "오늘 너무 힘들어"
    probs = predict(text, model, tokenizer)
    print(probs)