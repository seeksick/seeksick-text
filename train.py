import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score

# 1. 감정 라벨 정의
label_names = ['행복', '우울', '놀람', '분노', '중립']
label2id = {name: i for i, name in enumerate(label_names)}
id2label = {i: name for name, i in label2id.items()}

# 2. 데이터셋 정의
class EmotionDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=64):
        df = pd.read_csv(csv_path)
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# 3. 모델 정의
class KoBERTClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 5)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {'loss': loss, 'logits': torch.softmax(logits, dim=1)}

# 4. 평가 지표
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted'),
    }

# 5. 학습 실행
def train():
    model_name = "monologg/kobert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = KoBERTClassifier.from_pretrained(model_name)

    train_dataset = EmotionDataset("data/train.csv", tokenizer)
    val_dataset = EmotionDataset("data/test.csv", tokenizer)

    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./models/kobert_emotion")
    print("모델 저장 완료!")

if __name__ == "__main__":
    train()