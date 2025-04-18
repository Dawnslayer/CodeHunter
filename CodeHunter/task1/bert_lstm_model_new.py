import torch
import torch.nn as nn
from transformers import BertModel,BertConfig,AutoTokenizer,AutoModel
import  os

class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTBiLSTMClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('../Salesforce/codet5p-110m-embedding', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained('../Salesforce/codet5p-110m-embedding')
        self.bilstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def getTokenizer(self):
        return self.tokenizer
    def forward(self, input_ids, attention_mask):
        batch_size, seq_count, max_seq_length = input_ids.size()

        # Reshape input_ids and attention_mask to combine batch dimension and sequence dimension
        input_ids = input_ids.view(-1, max_seq_length)
        attention_mask = attention_mask.view(-1, max_seq_length)

        # BERT encoding
        bert_encoded = self.bert(input_ids, attention_mask).last_hidden_state
        bert_encoded = bert_encoded.mean(dim=1)
        # Reshape bert_encoded back to original shape
        bert_encoded = bert_encoded.view(batch_size, -1, 768)

        # LSTM encoding
        lstm_output, _ = self.bilstm(bert_encoded)

        # Classification layer
        logits = self.fc(lstm_output)

        return logits
