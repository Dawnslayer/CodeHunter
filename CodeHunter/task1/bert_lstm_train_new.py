# coding=utf8
import logging

# from transformers import BertModel, BertConfig
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import metrics
from multiprocessing import freeze_support
import sys
import os
from ast_bert_data_process import predict_data_process
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from bert_lstm_model_new import BERTBiLSTMClassifier



detail_logger = logging.getLogger('info')
detail_logger.setLevel(logging.INFO)

detail_handler = logging.FileHandler('detail_logs.log')
detail_logger.addHandler(detail_handler)
detail_logger.info("starting.........")


class CodeDataset(Dataset):
    def __init__(self, path):
        data = pd.read_pickle(path)
        self.data = data

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = None
        try:
            item = self.data[idx]
            lines = item['lines']
            labels = item['labels']
            lengths = item['lengths']
            lines_count = item['lines_count']


            # lines = [token for func in lines for token in func]


            labels = torch.tensor(labels, dtype=torch.long)


            lines = torch.tensor(lines, dtype=torch.long)


            lengths = torch.tensor(lengths, dtype=torch.long)
            return {"lines": lines, "labels": labels, "attention_mask": lengths, "lines_count": lines_count}
        except Exception as e:
            return {"lines": torch.zeros([50, 20], dtype=torch.long), "labels": torch.zeros([50], dtype=torch.long), "attention_mask": torch.zeros([50, 20], dtype=torch.long), "lines_count": lines_count}


def evaluate(model, data_loader, device):
    criterion = nn.CrossEntropyLoss().to(device)

    model.eval()
    y_true, y_pred = [], []
    flags = []
    total_eval_loss = 0
    total_eval_accuracy = 0
    total_eval_f1 = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc='VALID')):
            lines_batch = batch['lines'].to(device)
            labels_batch = batch['labels'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            lines_count_batch = batch['lines_count'].to(device)
            outputs = model(lines_batch, attention_mask=attention_mask_batch)
            output_flat = outputs.view(-1, 2)
            target_flat = labels_batch.view(-1)
            loss = criterion(output_flat, target_flat)
            total_eval_loss += loss.item()
            _, predicted_labels_batch = torch.max(outputs, dim=2)
            predicted_labels_batch = predicted_labels_batch.tolist()
            labels_batch = labels_batch.tolist()
            for j in range(lines_batch.size(0)):
                predicted_labels = predicted_labels_batch[j]
                label_ids = labels_batch[j]
                y_true.extend(label_ids)
                y_pred.extend(predicted_labels)

                total_eval_accuracy += flat_accuracy(predicted_labels, label_ids)
                total_eval_f1 += recall_score(predicted_labels, label_ids)
                count = count + 1
                flag = 0
                lines_count = lines_count_batch[j]
                length = lines_count
                if flat_accuracy_match(predicted_labels, label_ids, length):
                    flag = 1
                flags.append(flag)

    avg_val_accuracy = total_eval_accuracy / count
    avg_val_f1 = total_eval_f1 / count

    acc = sum(flags) / len(flags)
    print(
        'new Loss: {:.4f}, Val Acc: {:.4f}, LINE Acc: {:.4f}, LINE RECALL: {:.4f},METRICS precision_score: {:.4f},METRICS recall_score: {:.4f},METRICS f1_score: {:.4f}, sum_flags: {:.4f}, len_flags: {:.4f}'.format(
            total_eval_loss / count,
            acc, avg_val_accuracy, avg_val_f1,metrics.precision_score(y_true, y_pred),metrics.recall_score(y_true, y_pred),metrics.f1_score(y_true, y_pred), sum(flags), len(flags)))
    detail_logger.info(
        'new Loss: {:.4f}, Val Acc: {:.4f}, LINE Acc: {:.4f}, LINE RECALL: {:.4f},METRICS precision_score: {:.4f},METRICS recall_score: {:.4f},METRICS f1_score: {:.4f}, sum_flags: {:.4f}, len_flags: {:.4f}'.format(
            total_eval_loss / count,
            acc, avg_val_accuracy, avg_val_f1,metrics.precision_score(y_true, y_pred),metrics.recall_score(y_true, y_pred),metrics.f1_score(y_true, y_pred), sum(flags), len(flags)))
    return acc

def flat_accuracy(preds, labels):
    return metrics.accuracy_score(labels, preds)

def recall_score(preds, labels):
    pred_flat = preds
    labels_flat = labels
    return metrics.recall_score(labels_flat, pred_flat, zero_division = 1)


def flat_accuracy_match(preds, labels, length = 50):
    pred_flat = preds
    labels_flat = labels
    pred_flat = pred_flat[:length]
    labels_flat = labels_flat[:length]
    res = pred_flat == labels_flat
    return res


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    predictions = []
    true_labels = []
    for i, batch in enumerate(tqdm(train_loader, desc='Train')):
        lines_batch = batch['lines']
        labels_batch = batch['labels']
        attention_mask_batch = batch['attention_mask']
        lines_batch = lines_batch.to(device)
        labels_batch = labels_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        outputs = model(lines_batch, attention_mask=attention_mask_batch)
        output_flat = outputs.view(-1, 2)
        target_flat = labels_batch.view(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()

        if i % 16 == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        _, predicted_labels = torch.max(outputs, dim=2)
        predictions.extend(predicted_labels.tolist())
        true_labels.extend(labels_batch.tolist())

    avg_loss = total_loss / len(train_loader)
    accuracy = 0
    for i in range(len(true_labels)):
        accuracy = accuracy + flat_accuracy_match(true_labels[i], predictions[i])
    accuracy = accuracy / len(true_labels)
    return avg_loss, accuracy


def train_model(start_epoch, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # train_data = CodeDataset('processed_train.pkl')
    # valid_data = CodeDataset('processed_valid.pkl')

    # train_data = CodeDataset('processed_train_wo_ast.pkl')
    # valid_data = CodeDataset('processed_valid_wo_ast.pkl')

    train_data = CodeDataset('processed_train_ast.pkl')
    valid_data = CodeDataset('processed_valid_ast.pkl')
    test_data = CodeDataset('processed_test_ast.pkl')



    batch_size = 4
    workers = 1
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=workers)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                                  num_workers=workers)


    model = BERTBiLSTMClassifier(num_classes=2)
    if start_epoch == None or start_epoch < 0:
        start_epoch = 0
    model.to(device)
    if start_epoch > 0:
        filename = f'checkpoints/processed_test_ast_%d.pth.tar' % start_epoch
        model_state_dict = torch.load(filename)
        model.load_state_dict(model_state_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.bilstm.parameters(), 'lr': 1e-3}
    ])

    for epoch in range(start_epoch, num_epochs):
        avg_loss, accuracy = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        filename = f'checkpoints/processed_test_ast_%d.pth.tar' % epoch
        torch.save(model.state_dict(), filename)
        detail_logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        evaluate(model, valid_dataloader, device)

def predict(model, device, data):

    line_limit = 50
    word_max_length = 64
    code_lines = data.split("\n")
    func_dict = predict_data_process(model, code_lines, line_limit, word_max_length)
    lines = func_dict['lines']
    mask = func_dict['mask']

    lines = torch.tensor(lines, dtype=torch.long)

    mask = torch.tensor(mask, dtype=torch.long)
    lines = lines.unsqueeze(0)
    mask = mask.unsqueeze(0)


    lines_batch = lines.to(device)
    attention_mask_batch = mask.to(device)
    outputs = model(lines_batch, attention_mask=attention_mask_batch)
    _, predicted_labels_batch = torch.max(outputs, dim=2)
    predicted_labels_batch = predicted_labels_batch.squeeze()
    predicted_labels_batch = predicted_labels_batch.tolist()
    return predicted_labels_batch



if __name__ == '__main__':
    freeze_support()
    mode = sys.argv[1]
    if mode == 'train':
        train_model(0, 50)
    elif mode == 'test':
        epoch = sys.argv[2]
        batch_size = 4
        workers = 1
        test_data = CodeDataset('processed_test_ast.pkl')
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=workers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BERTBiLSTMClassifier(num_classes=2)
        filename = f'checkpoints/bert_bilstm_ast_%d.pth.tar' % int(epoch)
        model_state_dict = torch.load(filename)
        model.load_state_dict(model_state_dict)
        model.to(device)
        evaluate(model, test_dataloader, device)
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
        res = predict(data)
        print(res)



