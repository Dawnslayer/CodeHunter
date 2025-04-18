import torch.nn as nn
from transformers import BertModel
from torch.utils.data import DataLoader
import logging
from torch.utils.data import Dataset
from transformers import BertTokenizer
import sys
from multiprocessing import freeze_support
import torch
from sklearn import metrics
from prepare_bert import predict_data_process


logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class CustomDataset(Dataset):
    def __init__(self, in_try_file, before_try_file,  label_file, max_length):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

        with open(in_try_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            self.in_try_texts = [line.strip() for line in lines]

        with open(before_try_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            self.before_try_texts = [line.strip() for line in lines]

        with open(label_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            self.labels = [int(line.strip()) for line in lines]

    def __len__(self):
        return len(self.in_try_texts)

    def __getitem__(self, idx):
        in_try_text = self.in_try_texts[idx]
        before_try_text = self.before_try_texts[idx]
        label = self.labels[idx]


        inputs_in_try = self.tokenizer(in_try_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs_before_try = self.tokenizer(before_try_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return inputs_in_try['input_ids'].squeeze(),\
               inputs_in_try['attention_mask'].squeeze(), \
               inputs_before_try['input_ids'].squeeze(), \
               inputs_before_try['attention_mask'].squeeze(), \
               torch.tensor(label, dtype=torch.long)

class CustomModel(nn.Module):
    def __init__(self, num_classes, bert_model, lstm_hidden_size):
        super(CustomModel, self).__init__()


        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        self.lstm_before_try = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.lstm_in_try = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, batch_first=True, bidirectional=True)


        self.fc = nn.Linear(lstm_hidden_size * 4, num_classes)
        self.activation = nn.ReLU()

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try):

        outputs_before = self.bert(input_ids=input_ids_before, attention_mask=attention_mask_before)
        sequence_output_before = outputs_before.last_hidden_state

        outputs_in_try = self.bert(input_ids=input_ids_in_try, attention_mask=attention_mask_in_try)
        sequence_output_in_try = outputs_in_try.last_hidden_state


        _, (lstm_hidden_before, _) = self.lstm_before_try(sequence_output_before)
        _, (lstm_hidden_in_try, _) = self.lstm_in_try(sequence_output_in_try)


        combined_hidden = torch.cat((lstm_hidden_before[0], lstm_hidden_before[1], lstm_hidden_in_try[0], lstm_hidden_in_try[1]), dim=1)

        output = self.fc(combined_hidden)
        output = self.activation(output)

        return output

def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch in data_loader:
            input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try, labels = batch
            input_ids_before = input_ids_before.to(device)
            attention_mask_before = attention_mask_before.to(device)
            input_ids_in_try = input_ids_in_try.to(device)
            attention_mask_in_try = attention_mask_in_try.to(device)
            labels = labels.to(device)
            outputs = model(input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds, average='weighted', zero_division = 1)
    recall = metrics.recall_score(all_labels, all_preds, average='weighted', zero_division = 1)
    f1 = metrics.f1_score(all_labels, all_preds, average='weighted', zero_division = 1)

    return accuracy, precision, recall, f1



def evaluate_topn(model, data_loader, device, n=1):
    model.eval()
    with torch.no_grad():
        total_samples = 0
        correct_samples = 0
        for batch in data_loader:
            input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try, labels = batch
            input_ids_before = input_ids_before.to(device)
            attention_mask_before = attention_mask_before.to(device)
            input_ids_in_try = input_ids_in_try.to(device)
            attention_mask_in_try = attention_mask_in_try.to(device)
            labels = labels.to(device)
            outputs = model(input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try)


            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_n_indices = torch.topk(probabilities, k=n).indices


            preds = top_n_indices[:, :n]
            is_correct = torch.any(preds == labels.unsqueeze(1), dim=1)
            total_samples += len(labels)
            correct_samples += is_correct.sum().item()

    topn_accuracy = correct_samples / total_samples
    return topn_accuracy


def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                result_dict[key] = value
    return result_dict

def train(version):
    if version == None:
        version = 'nexgen_ast'
    if version not in ['nexgen_ast', 'nexgen', 'drex']:
        exit(-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 50
    batch_size = 16
    num_classes = 10
    bert_model = 'bert-base-uncased'
    lstm_hidden_size = 256

    multi_slicing = 'data/multi_slicing/'
    train_data = CustomDataset(multi_slicing + 'src-nexgen-ast-train.back',
                               multi_slicing + 'src-nexgen-ast-train.front',
                               multi_slicing + 'nexgen-ast-train-label.txt', max_length=256)
    valid_data = CustomDataset(multi_slicing + 'src-nexgen-ast-valid.back',
                               multi_slicing + 'src-nexgen-ast-valid.front',
                               multi_slicing + 'nexgen-ast-valid-label.txt', max_length=256)
    if version == 'drex':
        num_classes = 52
        multi_slicing = 'data/drex/'
        train_data = CustomDataset(multi_slicing + 'src-train.back', multi_slicing + 'src-train.front', multi_slicing + 'train-label.txt', max_length=256)
        valid_data = CustomDataset(multi_slicing + 'src-valid.back', multi_slicing + 'src-valid.front', multi_slicing + 'valid-label.txt', max_length=256)
    if version == 'nexgen':
        train_data = CustomDataset(multi_slicing + 'src-nexgen-train.back',
                                   multi_slicing + 'src-nexgen-train.front',
                                   multi_slicing + 'nexgen-train-label.txt', max_length=256)
        valid_data = CustomDataset(multi_slicing + 'src-nexgen-valid.back',
                                   multi_slicing + 'src-nexgen-valid.front',
                                   multi_slicing + 'nexgen-valid-label.txt', max_length=256)

    workers = 4

    model = CustomModel(num_classes=num_classes, bert_model=bert_model, lstm_hidden_size=lstm_hidden_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.lstm_before_try.parameters(), 'lr': 1e-3},
        {'params': model.lstm_in_try.parameters(), 'lr': 1e-3}
    ])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=workers)


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for i, batch in enumerate(train_loader):

            input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try, labels = batch
            input_ids_before = input_ids_before.to(device)
            attention_mask_before = attention_mask_before.to(device)
            input_ids_in_try = input_ids_in_try.to(device)
            attention_mask_in_try = attention_mask_in_try.to(device)
            labels = labels.to(device)
            outputs = model(input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try)
            loss = criterion(outputs, labels)

            loss.backward()


            if i % 4 == 0:
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
                      f"Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
                total_loss = 0.0
                correct = 0
                total = 0


        valid_accuracy, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, device)


        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {valid_accuracy:.4f}, "
                     f"Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, "
                     f"F1-Score: {valid_f1:.4f}")
        filename = f'checkpoints/bert_bilstm_for_catch_nexgen_ast_%d.pth.tar' % epoch
        if version == 'drex':
            filename = f'checkpoints/bert_bilstm_for_catch_drex_%d.pth.tar' % epoch
        elif  version == 'nexgen':
            filename = f'checkpoints/bert_bilstm_for_catch_nexgen_%d.pth.tar' % epoch
        torch.save(model.state_dict(), filename)

def predict(model, device, code_lines, begin_position, end_position):


    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = model.get_tokenizer()
    max_length = 256

    before_try_text, in_try_text = predict_data_process(code_lines, begin_position, end_position)

    inputs_in_try = tokenizer(in_try_text, padding='max_length', truncation=True, max_length=max_length,
                                   return_tensors="pt")
    inputs_before_try = tokenizer(before_try_text, padding='max_length', truncation=True,
                                       max_length=max_length, return_tensors="pt")

    input_ids_before = inputs_before_try['input_ids']
    attention_mask_before = inputs_before_try['attention_mask']
    input_ids_in_try = inputs_in_try['input_ids']
    attention_mask_in_try =  inputs_in_try['attention_mask']
    input_ids_before = input_ids_before.to(device)
    attention_mask_before = attention_mask_before.to(device)
    input_ids_in_try = input_ids_in_try.to(device)
    attention_mask_in_try = attention_mask_in_try.to(device)
    outputs = model(input_ids_before, attention_mask_before, input_ids_in_try, attention_mask_in_try)
    outputs = outputs.squeeze()

    probabilities = torch.nn.functional.softmax(outputs, dim=0)
    top_n_indices = torch.topk(probabilities, k=10).indices
    type = top_n_indices[0]
    type = str(int(type))
    type_dict = read_txt_to_dict('./exception-class-map-reverse.txt')


    return type_dict[type]





if __name__ == '__main__':
    freeze_support()
    mode = sys.argv[1]
    if mode == 'train':
        version = 'nexgen_ast'
        if len(sys.argv) >= 2 or sys.argv[2] !=  None or sys.argv[2] !=  None:
            version = sys.argv[2] # nexgen_ast、nexgen、drex
        train(version)
    elif mode == 'test_topn':
        epoch = sys.argv[2]
        version = 'nexgen_ast'
        if len(sys.argv) >= 3 or sys.argv[3] !=  None or sys.argv[3] !=  None:
            version = sys.argv[3] # nexgen_ast、nexgen、drex
        batch_size = 32
        workers = 1
        # num_classes = 52
        num_classes = 10
        # multi_slicing = 'data/drex/'
        multi_slicing = 'data/multi_slicing/'
        filename = f'checkpoints/bert_bilstm_for_catch_nexgen_ast_%d.pth.tar' % int(epoch)
        valid_data = CustomDataset(multi_slicing + 'src-nexgen-ast-test.back', multi_slicing + 'src-nexgen-ast-test.front',
                                   multi_slicing + 'nexgen-ast-test-label.txt', max_length=256)
        if version == 'drex':
            multi_slicing = 'data/drex/'
            valid_data= CustomDataset(multi_slicing + 'src-test.back', multi_slicing + 'src-test.front',
                                   multi_slicing + 'test-label.txt', max_length=256)
            num_classes = 52
            filename = f'checkpoints/bert_bilstm_for_catch_drex_%d.pth.tar' % int(epoch)
        if version == 'nexgen':
            valid_data = CustomDataset(multi_slicing + 'src-nexgen-test.back',
                                       multi_slicing + 'src-nexgen-test.front',
                                       multi_slicing + 'nexgen-test-label.txt', max_length=256)
            filename = f'checkpoints/bert_bilstm_for_catch_nexgen_%d.pth.tar' % int(epoch)


        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=workers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bert_model = 'bert-base-cased'
        bert_model = 'bert-base-uncased'

        lstm_hidden_size = 256
        model = CustomModel(num_classes=num_classes, bert_model=bert_model, lstm_hidden_size=lstm_hidden_size)
        model_state_dict = torch.load(filename)
        model.load_state_dict(model_state_dict)
        model.to(device)

        top1 = evaluate_topn(model, valid_loader, device, 1)
        top2 = evaluate_topn(model, valid_loader, device, 2)
        top3 = evaluate_topn(model, valid_loader, device, 3)
        top5 = evaluate_topn(model, valid_loader, device, 5)
        top10 = evaluate_topn(model, valid_loader, device, 10)
        logging.info(f"evaluate_topn Epoch {epoch} - Accuracy top1: {top1:.4f}, "
                     f"top2: {top2:.4f}, top2: {top3:.4f}, "
                     f"top5: {top5:.4f}, "
                     f"top10: {top10:.4f}, ")
    elif mode == 'predict':
        data = '''public static void unGunzipFile ( String compressedFile , String decompressedFile ) {
                byte [ ] buffer = new byte [ 1024 ] ;
                FileSystem fs = FileSystem . getLocal ( new Configuration ( ) ) ;
                FSDataInputStream fileIn = fs . open ( new Path ( compressedFile ) ) ;
                GZIPInputStream gZIPInputStream = new GZIPInputStream ( fileIn ) ;
                FileOutputStream fileOutputStream = new FileOutputStream ( decompressedFile ) ;
                int bytes_read ;
                while ( ( bytes_read = gZIPInputStream . read ( buffer ) ) > 0 ) {
                    fileOutputStream . write ( buffer , 0 , bytes_read ) ;
                }
                gZIPInputStream . close ( ) ;
                fileOutputStream . close ( ) ;
                }'''
        print(predict(data, 2, 11))
