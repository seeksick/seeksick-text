from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from dataset import EmotionDataset
from model import KoBERTForEmotion
import torch
from data_processor import process_kemdy20_data

def train():
    # 1. 데이터 전처리
    train_df, val_df, test_df = process_kemdy20_data()
    
    # 2. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    
    # 3. 데이터셋 생성
    train_dataset = EmotionDataset(train_df, tokenizer)
    val_dataset = EmotionDataset(val_df, tokenizer)
    test_dataset = EmotionDataset(test_df, tokenizer)
    
    # 4. 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 5. 모델 초기화
    model = KoBERTForEmotion.from_pretrained("monologg/kobert")
    
    # 6. 학습 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 7. 학습 루프
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # 8. 모델 저장
    torch.save(model.state_dict(), 'kobert_emotion_model.pt')

if __name__ == "__main__":
    train()