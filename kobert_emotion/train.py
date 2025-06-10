import os
import sys
import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from kobert_emotion.dataset import EmotionDataset
from kobert_emotion.model import KoBERTForEmotion
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_log.txt'),
        logging.StreamHandler(sys.stdout)
    ],
    encoding='utf-8'  # 한글 인코딩 깨짐 방지
)

# 모델 저장 경로 설정
MODEL_SAVE_PATH = 'models/best_model.pt'

def calculate_metrics(y_true, y_pred):
    # 다중 레이블 분류를 위한 메트릭스 계산
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    # 입력이 2차원이면 axis=1, 1차원이면 그냥 mean
    if len(y_true.shape) == 2:
        accuracy = np.mean(np.all(y_true == y_pred, axis=1))
    else:
        accuracy = np.mean(y_true == y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train(happy_limit=100, neutral_limit=100):
    try:
        logging.info("학습 시작")
        
        # 데이터 로드
        train_df = pd.read_csv('data/train.csv')
        train_aug = pd.read_csv('data/train_aug.csv')
        train_df = pd.concat([train_df, train_aug], ignore_index=True)
        # happy, neutral 샘플 수 제한
        happy_df = train_df[train_df['happy'] == 1].sample(n=happy_limit, random_state=42)
        neutral_df = train_df[train_df['neutral'] == 1].sample(n=neutral_limit, random_state=42)
        other_df = train_df[(train_df['happy'] != 1) & (train_df['neutral'] != 1)]
        train_df = pd.concat([happy_df, neutral_df, other_df], ignore_index=True)
        # test_ko_emotion.csv는 더 이상 train에 포함하지 않음
        # val/test에 aug 항상 추가
        val_data = pd.read_csv('data/val.csv')
        val_aug = pd.read_csv('data/val_aug.csv')
        val_data = pd.concat([val_data, val_aug], ignore_index=True)
        test_data = pd.read_csv('data/test.csv')
        test_aug = pd.read_csv('data/test_aug.csv')
        test_data = pd.concat([test_data, test_aug], ignore_index=True)
        logging.info(f'train: happy/neutral 샘플 제한, 총 {len(train_df)}개')
        logging.info(f'val: val_aug 포함, 총 {len(val_data)}개')
        logging.info(f'test: test_aug 포함, 총 {len(test_data)}개')
        
        # 텍스트에서 쌍따옴표 제거
        train_df['text'] = train_df['text'].str.replace('"', '')
        val_data['text'] = val_data['text'].str.replace('"', '')
        test_data['text'] = test_data['text'].str.replace('"', '')
        
        # NaN을 0으로 채우고 int로 변환
        for col in ['happy', 'depressed', 'surprised', 'angry', 'neutral']:
            train_df[col] = train_df[col].fillna(0).astype(int)
            val_data[col] = val_data[col].fillna(0).astype(int)
            test_data[col] = test_data[col].fillna(0).astype(int)
        
        # GPU 사용 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"학습 디바이스: {device}")
        
        # 각 감정별 positive 샘플 수 로깅 및 pos_weight 계산
        emotion_columns = ['happy', 'depressed', 'surprised', 'angry', 'neutral']
        for emotion in emotion_columns:
            positive_count = len(train_df[train_df[emotion] == 1])
            logging.info(f"{emotion} positive samples: {positive_count}")
        
        # pos_weight 계산 (음성/양성 비율)
        sample_counts = [train_df[emotion].sum() for emotion in emotion_columns]
        total = len(train_df)
        pos_weights = [(total - c) / c if c > 0 else 1.0 for c in sample_counts]
        # device에 맞게 tensor 변환
        pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)
        
        # 토크나이저 초기화
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
        logging.info("토크나이저 로드 완료")
        
        # 데이터셋 생성
        train_dataset = EmotionDataset(train_df, tokenizer)
        val_dataset = EmotionDataset(val_data, tokenizer)
        test_dataset = EmotionDataset(test_data, tokenizer)
        logging.info("데이터셋 생성 완료")
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        logging.info("데이터로더 생성 완료")
        
        # 모델 초기화
        model = KoBERTForEmotion()
        model.to(device)
        logging.info("모델 초기화 완료")
        
        # 옵티마이저와 손실 함수 설정
        optimizer = AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Learning Rate Scheduler 설정
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        
        # Early stopping 설정
        best_f1 = 0
        patience = 3
        patience_counter = 0
        EPOCHS = 5
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
            
            # 검증
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # 평균 메트릭 계산 (single-label)
            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)
            avg_metrics = calculate_metrics(all_labels_np, all_preds_np)
            logging.info("Validation - Average Metrics:")
            logging.info(f"Accuracy: {avg_metrics['accuracy']:.4f}, Precision: {avg_metrics['precision']:.4f}, "
                       f"Recall: {avg_metrics['recall']:.4f}, F1: {avg_metrics['f1']:.4f}")
            
            # 클래스별 메트릭 계산
            for i, emotion in enumerate(['happy', 'depressed', 'surprised', 'angry', 'neutral']):
                metrics = calculate_metrics((all_labels_np == i).astype(int), (all_preds_np == i).astype(int))
                logging.info(f"{emotion.capitalize()} - Accuracy: {metrics['accuracy']:.4f}, "
                           f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                           f"F1: {metrics['f1']:.4f}")
            
            # Early stopping 체크
            current_f1 = avg_metrics['f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                logging.info(f"Best model saved (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                logging.info(f"No improvement for {patience_counter} epochs")
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Learning Rate Scheduler 스텝
            scheduler.step(current_f1)
            logging.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        
        # 최종 테스트
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 최종 테스트 결과 출력
        logging.info("\nFinal Test Results:")
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        avg_metrics = calculate_metrics(all_labels_np, all_preds_np)
        logging.info(f"Average Metrics - Accuracy: {avg_metrics['accuracy']:.4f}, "
                    f"Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}, "
                    f"F1: {avg_metrics['f1']:.4f}")
        
        for i, emotion in enumerate(['happy', 'depressed', 'surprised', 'angry', 'neutral']):
            metrics = calculate_metrics((all_labels_np == i).astype(int), (all_preds_np == i).astype(int))
            logging.info(f"{emotion.capitalize()} - Accuracy: {metrics['accuracy']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                       f"F1: {metrics['f1']:.4f}")
        
        # val/test에 happy, neutral 샘플 제한
        happy_val = val_data[val_data['happy'] == 1].sample(n=min(100, len(val_data[val_data['happy'] == 1])), random_state=42)
        neutral_val = val_data[val_data['neutral'] == 1].sample(n=min(100, len(val_data[val_data['neutral'] == 1])), random_state=42)
        other_val = val_data[(val_data['happy'] != 1) & (val_data['neutral'] != 1)]
        val_data = pd.concat([happy_val, neutral_val, other_val], ignore_index=True)

        happy_test = test_data[test_data['happy'] == 1].sample(n=min(100, len(test_data[test_data['happy'] == 1])), random_state=42)
        neutral_test = test_data[test_data['neutral'] == 1].sample(n=min(100, len(test_data[test_data['neutral'] == 1])), random_state=42)
        other_test = test_data[(test_data['happy'] != 1) & (test_data['neutral'] != 1)]
        test_data = pd.concat([happy_test, neutral_test, other_test], ignore_index=True)
        
    except Exception as e:
        logging.error(f"에러 발생: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(
        filename='train_log.txt',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        encoding='utf-8'  # 한글 인코딩 깨짐 방지
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--happy_limit', type=int, default=100, help='happy 샘플 최대 개수')
    parser.add_argument('--neutral_limit', type=int, default=100, help='neutral 샘플 최대 개수')
    args = parser.parse_args()
    train(happy_limit=args.happy_limit, neutral_limit=args.neutral_limit)