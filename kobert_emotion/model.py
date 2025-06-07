from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn

class KoBERTForEmotion(BertPreTrainedModel):
    def __init__(self, config, num_labels=5):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = self.softmax(logits)
        loss = None
        if labels is not None:
            loss_fn = nn.KLDivLoss(reduction='batchmean')
            loss = loss_fn(torch.log(probs + 1e-8), labels)
        return {'loss': loss, 'logits': probs}