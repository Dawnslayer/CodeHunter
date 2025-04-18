from flask import Flask, request, jsonify
import torch
from task1 import bert_lstm_train_new,bert_lstm_model_new
from task2 import bert_train_new
import data_access


app = Flask(__name__)


# init model
# task1
task1_model = bert_lstm_model_new.BERTBiLSTMClassifier(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filename = f'../task1/checkpoints/bert_bilstm_replace_var_49.pth.tar'
model_state_dict = torch.load(filename, map_location=device)
task1_model.load_state_dict(model_state_dict)
task1_model.to(device)

# task2
bert_model = 'bert-base-uncased'
lstm_hidden_size = 256
num_classes = 10
task2_model = bert_train_new.CustomModel(num_classes=num_classes, bert_model=bert_model, lstm_hidden_size=lstm_hidden_size)
filename = f'../task2/checkpoints/bert_bilstm_for_catch_nexgen_49.pth.tar'
model_state_dict = torch.load(filename, map_location=device)
task2_model.load_state_dict(model_state_dict)
task2_model.to(device)




@app.route('/predict.json', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json(force=True)
        code_data = data['codeData']
        error_code = 0

        lines_predict_res = bert_lstm_train_new.predict(task1_model, device, code_data)
        begin_position, end_position = find_begin_end_position(lines_predict_res)


        exception_type = 'Exception'
        predict_exception_type = bert_train_new.predict(task2_model, device, code_data, begin_position, end_position)
        if (predict_exception_type != None):
            exception_type = predict_exception_type
        try:
            lines_predict_res_str = ",".join(list(map(str, lines_predict_res)))
            data_access.insert_code_hunter_record(code_data, lines_predict_res_str, exception_type, -1, '', '')
        except Exception as e:
            print("insert_code_hunter_record error", e)
        response = {}
        response['exceptionLinesBegin'] = begin_position
        response['exceptionLinesEnd'] = end_position
        response['exceptionType'] = exception_type
        response['errorCode'] = error_code
        return jsonify(response)


def find_begin_end_position(lines_predict_res):
    begin_position = -1
    for i in range(len(lines_predict_res)):
        if(lines_predict_res[i] == 1 and begin_position < 0):
            begin_position = i
        if(lines_predict_res[i] == 0 and begin_position > 0):
            end_position = i - 1
            return begin_position, end_position
    return begin_position, len(lines_predict_res)


if __name__ == '__main__':
    app.run(debug=True)
