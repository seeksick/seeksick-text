import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer=None, max_length=128):
        self.texts = df['text'].values.tolist()
        self.labels = df[['happy', 'depressed', 'surprised', 'angry', 'neutral']].values.argmax(axis=1).astype('int64')
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }