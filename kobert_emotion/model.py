import torch
import torch.nn as nn
from transformers import AutoModel

class KoBERTForEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.kobert = AutoModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 5)  # 5개의 감정 클래스 (softmax 분류)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.kobert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[0][:, 0, :]  # [CLS] 토큰의 임베딩
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits  # softmax 분류용 로짓 반환