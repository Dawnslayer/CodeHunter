# Task1 (Exception Localization)
## Introduction
Used to locate code that may have exceptions, i.e. try block localization.

## Nexgen Experiments
1. Download the compressed file task1_data.zip from [https://figshare.com/s/e25c96203125cfd10749](https://figshare.com/s/e25c96203125cfd10749) and unzip it to the task1/data directory.
2. Run `python ast_bert_data_process.py` to preprocess data.
3. Run `python bert_lstm_train_new.py train` to train the model.
4. Run `python bert_lstm_train_new.py test 40` to test model.

## Neurex Experiments
1. Download the compressed file Neurex_data.zip from [https://figshare.com/s/d83dbedb9359ff79a670](https://figshare.com/s/d83dbedb9359ff79a670) and unzip it to the neurex/processed directory.
2. Run `python model_train_task1.py train` to train the model.
3. Run `python model_train_task1.py test` to test model.


# Task2 (Exception Type Prediction)
## Introduction
Used to predict the types of exceptions that may arise from possible exceptional code lines.

## Nexgen and D-REX Experiments
1. Download the compressed file task2_data.zip from [https://figshare.com/s/d51fb4abb800959ed8a3](https://figshare.com/s/d51fb4abb800959ed8a3) and unzip it to the task2/data directory.
2. Run `python prepare_bert.py` to preprocess data.
3. Run `python bert_train_new.py train` to train the model.
4. Run `python bert_train_new.py test_topn 40` to test model.

## Neurex Experiments
1. Download the compressed file Neurex_data.zip from [https://figshare.com/s/d83dbedb9359ff79a670](https://figshare.com/s/d83dbedb9359ff79a670) and unzip it to the neurex/processed directory.
2. Run `python model_train_task2.py train` to train the model.
3. Run `python model_train_task2.py test` to test model.


# IDE Plugin of CodeHunter
## Introduction
users can use the functions provided by the IDE plugin to detect and handle exceptions in the program code automatically.

## How to run?
1. Open CodeHunterPlugin By IntelliJ IDEA.
2. Move the model parameter files for task1 and task2 to the plugin/checkpoints directory. You can download the model parameters from [https://figshare.com/s/aa53e8a6e1e0178493c8](https://figshare.com/s/aa53e8a6e1e0178493c8) or use the model parameters trained by yourselves.
3. Run `python plugin-web.py` to start the plugin back-end service.
