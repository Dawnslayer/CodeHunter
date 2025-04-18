import re

import javalang
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import traceback
from ast_bert_data_process_util import get_feature_tokens,get_feature_tokens_for_api
import logging
logging.basicConfig(filename='new_data_process.log', level=logging.DEBUG)

parse_error_count = 0

PAD = '[PAD]'


def tokenize_and_encode(code, tokenizer, word_max_length):
    if len(code) > word_max_length - 2:
        code = code[:word_max_length - 2]
    if (code == PAD):
        return [0] * word_max_length, [0] * word_max_length
    code = tokenizer.tokenize(code)
    if len(code) > word_max_length - 2:
        code = code[:word_max_length - 2]
    tokens = ['[CLS]'] + code + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = word_max_length - len(input_ids)

    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    return input_ids, input_mask


def limit_lines_length(lines, line_limit, word_max_length):
    PAD = '[PAD]'

    if(len(lines) >= line_limit):
        return lines[:line_limit]

    res = lines + [PAD] * (line_limit - len(lines))
    return res


def limit_tags_length(tags, line_limit):
    if(len(tags) >= line_limit):
        return tags[:line_limit]
    res = tags + [0] * (line_limit - len(tags))
    return res


def processed_data(path, tokenizer, word_max_length = 20, line_limit = 50):

    processed_data = []

    data = pd.read_pickle(path)
    # data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    for i in tqdm(range(data.shape[0])):
        row = data.iloc[i, :]
        code_lines, tags = row['lines'], row['labels']
        try:


            code_lines, tags = get_feature_tokens(code_lines, tags)
            lines_count = len(code_lines)
            if(len(code_lines) == 0):
                continue

            code_lines = limit_lines_length(code_lines, line_limit, word_max_length)
            tags = limit_tags_length(tags, line_limit)

            code_lines = [line.strip() for line in code_lines]

            code_lines = replace_line(code_lines)

            func_data = []
            for j, line in enumerate(code_lines):
                code = code_lines[j]
                label = tags[j]
                input_ids, length = tokenize_and_encode(code, tokenizer, word_max_length)
                func_data.append((input_ids, label, length))


            func_data = list(zip(*func_data))

            func_dict = {
                'lines': func_data[0],
                'labels': func_data[1],
                'lengths': func_data[2],
                'lines_count': lines_count
            }


            processed_data.append(func_dict)

        except Exception as e:

            print(e)
            continue
    logging.info("processed_data count:{}".format(len(processed_data)))
    return processed_data

def replace_line(lines):
    new_lines = []
    hit_words = {}
    for i in lines:
        temp = i.lstrip().replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')
        for k,v in hit_words.items():
            temp.replace(k, v)
        i_words = temp.lstrip().split(' ')

        if len(i_words) > 2 and (i_words[2] == '=' or i_words[2] == ';'):
            replacedWord = replaceHeadBig(i_words[0])
            hit_words[i_words[1]] = replacedWord
            i_words[1] = replacedWord
            temp = " ".join(i_words)
        new_lines.append(temp)
    return new_lines


def predict_data_process(model, data, line_limit, word_max_length):
    tags = [0] * len(data)
    code_lines, tags = get_feature_tokens_for_api(data, tags)
    lines_count = len(code_lines)
    if (len(code_lines) == 0):
        return {}

    code_lines = limit_lines_length(code_lines, line_limit, word_max_length)
    tags = limit_tags_length(tags, line_limit)

    code_lines = [line.strip() for line in code_lines]

    code_lines = replace_line(code_lines)
    func_data = []
    tokenizer = model.getTokenizer()
    for j, line in enumerate(code_lines):
        code = code_lines[j]
        label = tags[j]
        input_ids, length = tokenize_and_encode(code, tokenizer, word_max_length)
        func_data.append((input_ids, label, length))

    func_data = list(zip(*func_data))

    func_dict = {
        'lines': func_data[0],
        'labels': func_data[1],
        'mask': func_data[2],
        'lines_count': lines_count
    }
    return func_dict



def replaceHeadBig(word):
    if word == '' or word == None:
        return word
    return word[0].lower()+word[1:]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('../Salesforce/codet5p-110m-embedding')
    pd.to_pickle(processed_data('data/train.pkl', tokenizer, 64, 50), 'processed_train_ast.pkl')
    pd.to_pickle(processed_data('data/valid.pkl', tokenizer, 64, 50), 'processed_valid_ast.pkl')
    pd.to_pickle(processed_data('data/test.pkl', tokenizer, 64, 50), 'processed_test_ast.pkl')