import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):  # max_length 증가
        self.texts = df['text'].tolist()
        self.labels = df[['happy', 'depressed', 'surprised', 'angry', 'neutral']].values.astype(float)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # 문자열 변환 보장
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }