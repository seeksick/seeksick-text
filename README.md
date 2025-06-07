# seeksick-text

## 설치
pip install -r requirements.txt

## 학습
python -m kobert_emotion.train

## 추론
python -m kobert_emotion.infer

### 패키지 구조
```
seeksick-text/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── kobert_emotion/
│   ├── __init__.py
│   ├── dataset.py         # 데이터셋 클래스
│   ├── model.py           # KoBERT 기반 모델
│   ├── train.py           # 학습 스크립트
│   ├── infer.py           # 추론(예측) 스크립트
│   └── utils.py           # 유틸 함수(옵션)
│
├── requirements.txt
├── README.md
└── run_train.sh           # 학습 실행용 쉘 스크립트(옵션)
```