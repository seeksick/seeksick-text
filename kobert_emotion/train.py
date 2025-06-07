from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from dataset import EmotionDataset
from model import KoBERTForEmotion
import torch
from data_processor import process_kemdy20_data
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs['logits']
            
            preds = torch.argmax(probs, dim=1)
            labels = torch.argmax(labels, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train():
    # 1. 데이터 전처리
    train_df, val_df, test_df = process_kemdy20_data(
        data_dir="C:/Users/Server1/seeksick/KEMDy20_v1_2/annotation"
    )
    
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
    best_val_f1 = 0
    
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
        
        # 검증
        val_metrics = evaluate(model, val_loader, device)
        print(f'Validation Metrics - Accuracy: {val_metrics["accuracy"]:.4f}, F1: {val_metrics["f1"]:.4f}')
        
        # 최고 성능 모델 저장
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            os.makedirs('checkpoints', exist_ok=True)
            model.save_pretrained('checkpoints/best_model')
            tokenizer.save_pretrained('checkpoints/best_model')
    
    # 8. 최종 테스트
    test_metrics = evaluate(model, test_loader, device)
    print("\nFinal Test Results:")
    print(f'Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'Precision: {test_metrics["precision"]:.4f}')
    print(f'Recall: {test_metrics["recall"]:.4f}')
    print(f'F1 Score: {test_metrics["f1"]:.4f}')

if __name__ == "__main__":
    train()