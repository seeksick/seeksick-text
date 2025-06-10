import torch
import logging
import numpy as np
from transformers import BertTokenizer
# from model import KoBERTForEmotion  # (이 줄은 주석처리)
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 감정 클래스
EMOTIONS = ['happy', 'depressed', 'surprised', 'angry', 'neutral']

def predict_emotion(text):
    """
    주어진 텍스트의 감정을 예측합니다.
    Returns: (예측 감정, 감정별 확률 dict)
    """
    # 입력 텍스트 전처리
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    # 가장 높은 확률의 감정 선택
    pred_idx = np.argmax(probs)
    pred_emotion = EMOTIONS[pred_idx]
    # 감정별 확률 dict
    prob_dict = {emo: float(prob) for emo, prob in zip(EMOTIONS, probs)}
    return pred_emotion, prob_dict

def main():
    try:
        print("main 진입")
        sys.stdout.flush()
        test_sentences = [
            ("happy", "오늘 정말 기분이 좋아!"),
            ("depressed", "무기력하고 우울한 하루야"),
            ("surprised", "아악! 벌레가 나타났어"),
            ("angry", "왜 자꾸 안되는거야 진짜 짜증나"),
            ("neutral", "나 이제 집가는 중이야."),
        ]
        from transformers import AutoTokenizer
        import os
        import torch
        import numpy as np
        print("토크나이저 로드 시작")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
        print("토크나이저 로드 완료")
        sys.stdout.flush()
        # best_model.pt 경로 자동 탐색
        model_path = None
        for p in [os.path.join(os.path.dirname(__file__), "best_model.pt"), os.path.join(os.path.dirname(__file__), "../models/best_model.pt")]:
            if os.path.exists(p):
                model_path = p
                break
        print(f"모델 경로: {model_path}")
        sys.stdout.flush()
        if model_path is None:
            print("best_model.pt 파일을 찾을 수 없습니다.")
            sys.stdout.flush()
            exit(1)
        from kobert_emotion.model import KoBERTForEmotion
        print("모델 클래스 import 완료")
        sys.stdout.flush()
        model = KoBERTForEmotion()
        print("모델 인스턴스 생성 완료")
        sys.stdout.flush()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("모델 파라미터 로드 완료")
        sys.stdout.flush()
        model.eval()
        EMOTIONS = ["happy", "depressed", "surprised", "angry", "neutral"]
        for label, sent in test_sentences:
            try:
                print(f"문장 예측 시작: {sent}")
                sys.stdout.flush()
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
                with torch.no_grad():
                    logits = model(**inputs)
                    probs = torch.softmax(logits, dim=1)[0].numpy()
                pred_idx = np.argmax(probs)
                pred_emotion = EMOTIONS[pred_idx]
                print(f"문장: {sent}")
                print(f"실제 감정: {label}")
                print(f"예측 감정: {pred_emotion}")
                print(f"softmax 확률: {dict(zip(EMOTIONS, map(lambda x: round(float(x), 4), probs)))}\n")
                sys.stdout.flush()
            except Exception as e:
                print(f"문장: {sent}")
                print(f"예측 중 오류 발생: {e}\n")
                sys.stdout.flush()
    except Exception as e:
        print(f"main 전체 예외: {e}")
        sys.stdout.flush()

if __name__ == "__main__" or __name__.endswith(".predict"):
    main() 