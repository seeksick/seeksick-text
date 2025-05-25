import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

def run():
    # 1. 데이터 불러오기
    print("[INFO] 데이터 불러오는 중...")
    df = pd.read_csv('data/train.csv')  # 'text', 'label' 컬럼 필요
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # 2. KoBERT 토크나이저/모델 준비 (5개 클래스)
    print("[INFO] KoBERT 토크나이저와 모델 로딩...")
    tokenizer = BertTokenizer.from_pretrained('skt/kobert-base-v1')
    model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=5)
    
    # 3. Dataset 클래스 정의
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    
    dataset = NewsDataset(texts, labels)
    
    # 4. 트레이닝 아규먼트 (하이퍼파라미터 조정 가능)
    training_args = TrainingArguments(
        output_dir='./models',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=50,
        report_to="none"
    )
    
    # 5. Trainer 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    # 6. 학습 실행
    print("[INFO] 모델 학습 시작...")
    trainer.train()
    print("[INFO] 학습 완료! 모델이 ./models 아래 저장됨.")

if __name__ == "__main__":
    run()