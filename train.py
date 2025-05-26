import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
import torch.nn as nn
from tqdm import tqdm

# 1. 데이터셋 클래스
class EmotionDataset(Dataset):
    def __init__(self, csv_path, tok, vocab, max_len=64):
        df = pd.read_csv(csv_path)
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tok = tok
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tok(text)
        token_ids = self.vocab[tokens]
        # pad or truncate
        if len(token_ids) < self.max_len:
            token_ids += [1] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        valid_length = min(len(tokens), self.max_len)
        segment_ids = [0] * self.max_len
        return (torch.tensor(token_ids), torch.tensor(valid_length), torch.tensor(segment_ids), torch.tensor(label))

    def __len__(self):
        return len(self.labels)

# 2. 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=5, dr_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, token_ids, valid_length, segment_ids):
        _, pooled_output = self.bert(token_ids, valid_length, segment_ids)
        out = self.dropout(pooled_output)
        return self.classifier(out)

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # KoBERT 모델, vocab, 토크나이저 준비
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer_path = get_tokenizer()
    tok = SentencepieceTokenizer(tokenizer_path, num_best=0, alpha=0)

    # 데이터셋 및 DataLoader
    max_len = 64
    batch_size = 16
    train_dataset = EmotionDataset('data/train.csv', tok, vocab, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델, 손실함수, 옵티마이저
    model = BERTClassifier(bertmodel, num_classes=5, dr_rate=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 3

    print("[INFO] KoBERT 파인튜닝 시작!")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            token_ids, valid_length, segment_ids, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(token_ids, valid_length, segment_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    print("[INFO] 학습 완료! 모델을 저장합니다.")

    torch.save(model.state_dict(), "./models/kobert_emotion.pt")
    print("[INFO] 모델 저장 완료: ./models/kobert_emotion.pt")

if __name__ == "__main__":
    run()