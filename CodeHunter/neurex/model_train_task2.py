import torch
import torch.nn as nn
import logging
import sys

sys.path.append("../task2")

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from bert_train_new import CustomModel, evaluate
from tqdm import tqdm


detail_logger = logging.getLogger('info')
detail_logger.setLevel(logging.INFO)
detail_handler = logging.FileHandler('task2_train.log')
detail_logger.addHandler(detail_handler)


class CustomDatasetNeurex(Dataset):
    def __init__(self, in_try_file, before_try_file, label_file, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained('../Salesforce/codet5p-110m-embedding')
        self.max_length = max_length

        with open(in_try_file, 'r', encoding='UTF-8') as file:
            lines = file.readlines()
            self.in_try_texts = [line.strip() for line in lines]

        with open(before_try_file, 'r', encoding='UTF-8') as file:
            lines = file.readlines()
            self.before_try_texts = [line.strip() for line in lines]

        with open(label_file, 'r', encoding='UTF-8') as file:
            lines = file.readlines()
            self.labels = [int(line) for line in lines]

    def __len__(self):
        return len(self.in_try_texts)

    def __getitem__(self, idx):
        in_try_text = self.in_try_texts[idx]
        before_try_text = self.before_try_texts[idx]
        label = self.labels[idx]

        inputs_in_try = self.tokenizer(in_try_text, padding='max_length', truncation=True, max_length=self.max_length,
                                       return_tensors="pt")
        inputs_before_try = self.tokenizer(before_try_text, padding='max_length', truncation=True,
                                           max_length=self.max_length, return_tensors="pt")
        return inputs_in_try['input_ids'].squeeze(), \
            inputs_in_try['attention_mask'].squeeze(), \
            inputs_before_try['input_ids'].squeeze(), \
            inputs_before_try['attention_mask'].squeeze(), \
            torch.tensor(label, dtype=torch.long)


class Task2NeurexT5Model(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size):
        super(Task2NeurexT5Model, self).__init__()

        self.t5 = AutoModel.from_pretrained('../Salesforce/codet5p-110m-embedding', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained('../Salesforce/codet5p-110m-embedding')

        self.lstm_before_try = nn.LSTM(256, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.lstm_in_try = nn.LSTM(256, lstm_hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(lstm_hidden_size * 4, num_classes)
        self.activation = nn.ReLU()

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try):
        batch_size, seq_length = input_ids_before.size()

        outputs_before = self.t5(input_ids=input_ids_before, attention_mask=attention_mask_before)
        outputs_before = outputs_before.view(batch_size, 1, seq_length)

        outputs_in_try = self.t5(input_ids=input_ids_in_try, attention_mask=attention_mask_in_try)
        outputs_in_try = outputs_in_try.view(batch_size, 1, seq_length)

        _, (lstm_hidden_before, _) = self.lstm_before_try(outputs_before)
        _, (lstm_hidden_in_try, _) = self.lstm_in_try(outputs_in_try)

        combined_hidden = torch.cat((lstm_hidden_before[0], lstm_hidden_before[1], lstm_hidden_in_try[0], lstm_hidden_in_try[1]), dim=1)

        output = self.fc(combined_hidden)
        output = self.activation(output)

        return output


def train_model_neurex_t5(start_epoch, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_classes = 30
    lstm_hidden_size = 256

    train_data = CustomDatasetNeurex("processed/src_neurex_train.back", "processed/src_neurex_train.front",
                                     "processed/label_neurex_train.txt", max_length=256)
    valid_data = CustomDatasetNeurex("processed/src_neurex_valid_from_origin.back", "processed/src_neurex_valid_from_origin.front",
                                     "processed/label_neurex_valid_from_origin.txt", max_length=256)
    test_data = CustomDatasetNeurex("processed/src_neurex_test_from_origin.back", "processed/src_neurex_test_from_origin.front",
                                    "processed/label_neurex_test_from_origin.txt", max_length=256)
    model = Task2NeurexT5Model(num_classes=num_classes, lstm_hidden_size=lstm_hidden_size)
    model.to(device)
    if start_epoch is None or start_epoch < 0:
        start_epoch = 0
    if start_epoch > 0:
        filename = "checkpoints/neurex_ast_codet5_task2.pth.tar"
        model_state_dict = torch.load(filename)
        model.load_state_dict(model_state_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.t5.parameters(), 'lr': 2e-5},  # T5 底层层次的学习率
        {'params': model.lstm_before_try.parameters(), 'lr': 1e-3},  # BiLSTM 的学习率
        {'params': model.lstm_in_try.parameters(), 'lr': 1e-3}  # BiLSTM 的学习率
    ])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    acc = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for i, batch in enumerate(tqdm(train_loader, desc="Train")):
            input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try, labels = batch
            input_ids_before = input_ids_before.to(device)
            attention_mask_before = attention_mask_before.to(device)
            input_ids_in_try = input_ids_in_try.to(device)
            attention_mask_in_try = attention_mask_in_try.to(device)
            labels = labels.to(device)
            outputs = model(input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try)
            loss = criterion(outputs, labels)
            loss.backward()

            if i % 16 == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            total += labels.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()

            if (i + 1) % 100 == 0:
                average_loss = total_loss / 100
                accuracy = 100 * correct / total
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], "
                      f"Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.3f}%")
                total_loss = 0.0
                correct = 0
                total = 0
        valid_accuracy, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, device)
        detail_logger.info(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {valid_accuracy:.4f}, "
                           f"Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, "
                           f"F1-Score: {valid_f1:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {valid_accuracy:.4f}, "
              f"Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, "
              f"F1-Score: {valid_f1:.4f}")
        if valid_accuracy > acc:
            acc = valid_accuracy
            filename = "checkpoints/neurex_ast_codet5_task2.pth.tar"
            torch.save(model.state_dict(), filename)

    filename = "checkpoints/neurex_ast_codet5_task2_final.pth.tar"
    torch.save(model.state_dict(), filename)
    test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
    detail_logger.info(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
                       f"Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
          f"Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'train':
        train_model_neurex_t5(0, 30)
    elif mode == 'test':
        test_data = CustomDatasetNeurex("processed/src_neurex_test_from_origin.back",
                                        "processed/src_neurex_test_from_origin.front",
                                        "processed/label_neurex_test_from_origin.txt", max_length=256)
        test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Task2NeurexT5Model(num_classes=30, lstm_hidden_size=256)
        model.to(device)
        model_state_dict = torch.load("checkpoints/neurex_ast_codet5_task2.pth.tar")
        model.load_state_dict(model_state_dict)
        test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader, device)
        print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
              f"Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")
