import torch.nn as nn
import torch
import sys

sys.path.append("../task1")

from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer, AutoModel
from bert_lstm_train_new import CodeDataset, evaluate, train, detail_logger
from multiprocessing import freeze_support


class CodeT5BiLSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(CodeT5BiLSTMModel, self).__init__()
        self.t5 = AutoModel.from_pretrained('../Salesforce/codet5p-110m-embedding', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained('../Salesforce/codet5p-110m-embedding')
        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_count, max_seq_length = input_ids.size()

        # Reshape input_ids and attention_mask to combine batch dimension and sequence dimension
        input_ids = input_ids.view(-1, max_seq_length)
        attention_mask = attention_mask.view(-1, max_seq_length)
        # T5 encoding
        t5_encoded = self.t5(input_ids, attention_mask)
        # Reshape t5_encoded back to original shape
        t5_encoded = t5_encoded.view(batch_size, -1, 256)

        # LSTM encoding
        lstm_output, _ = self.bilstm(t5_encoded)

        # Classification layer
        logits = self.fc(lstm_output)

        return logits


def train_model_neurex_codet5(start_epoch, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = CodeDataset("processed/neurex_train_t5.pkl")
    valid_data = CodeDataset("processed/neurex_valid_t5_from_origin.pkl")
    test_data = CodeDataset("processed/neurex_test_t5_from_origin.pkl")
    batch_size = 4
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CodeT5BiLSTMModel(num_classes=2)
    if start_epoch is None or start_epoch < 0:
        start_epoch = 0
    model.to(device)
    if start_epoch > 0:
        filename = 'checkpoints/neurex_ast_codet5_task1.pth.tar'
        model_state_dict = torch.load(filename)
        model.load_state_dict(model_state_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.t5.parameters(), 'lr': 2e-5},
        {'params': model.bilstm.parameters(), 'lr': 1e-3}
    ])

    acc = 0
    for epoch in range(start_epoch, num_epochs):
        avg_loss, accuracy = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        detail_logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        cur_acc = evaluate(model, valid_dataloader, device)
        if cur_acc > acc:
            acc = cur_acc
            filename = "checkpoints/neurex_ast_codet5_task1.pth.tar"
            torch.save(model.state_dict(), filename)
            print(f"Epoch {epoch+1}: Valid Accuracy: {acc:.4f}")
            detail_logger.info(f"Epoch {epoch+1}: Valid Accuracy: {acc:.4f}")

    filename = "checkpoints/neurex_ast_codet5_task1_final.pth.tar"
    torch.save(model.state_dict(), filename)
    test_acc = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    detail_logger.info(f"Test Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'train':
        train_model_neurex_codet5(0, 30)
    elif mode == 'test':
        test_data = CodeDataset("processed/neurex_test_t5_from_origin.pkl")
        test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CodeT5BiLSTMModel(num_classes=2)
        model.to(device)
        model_state_dict = torch.load("checkpoints/neurex_ast_codet5_task1.pth.tar")
        model.load_state_dict(model_state_dict)
        test_acc = evaluate(model, test_dataloader, device)
        print(f"Test Accuracy: {test_acc:.4f}")
