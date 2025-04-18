import re
import os
import javalang
import json

from task2 import ast_bert_data_process_util_for_catch


def is_identifier(token):
    if re.match(r'\w+', token) and not re.match(r'\d+', token):
        if token not in javalang.tokenizer.Keyword.VALUES.union(javalang.tokenizer.BasicType.VALUES)\
                .union(javalang.tokenizer.Modifier.VALUES):
            return True
    return False

def get_try_index(code):
    start = -1
    stack = []
    for i, token in enumerate(code.split()):
        if token == 'try' and not stack:
            start = i
        elif token in ["'", '"']:
            if stack:
                if stack[-1] == token:
                    stack.pop()
            else:
                stack.append(token)
    return start

def get_try_index_for_ast(tokens):
    start = -1
    stack = []
    for i, token in enumerate(tokens):
        if token == 'TRY' and not stack:
            start = i
        elif token in ["'", '"']:
            if stack:
                if stack[-1] == token:
                    stack.pop()
            else:
                stack.append(token)
    return start

def get_catch_index_for_ast(tokens):
    start = -1
    stack = []
    for i, token in enumerate(tokens):
        if token == 'CatchClause' and not stack:
            start = i
        elif token in ["'", '"']:
            if stack:
                if stack[-1] == token:
                    stack.pop()
            else:
                stack.append(token)
    return start

def get_statements(code):
    tokens = code.split() if isinstance(code, str) else code
    intervals = []
    stack = []
    start = 0
    flag = False
    for i, token in enumerate(tokens):
        if token in ['"', "'"]:
            if stack:
                if stack[-1] == token:
                    stack.pop()
            else:
                stack.append(token)
            continue

        if not stack:
            if token in ['{', '}', ';'] and not flag:
                intervals.append((start, i))
                start = i+1
            elif token == '(':
                flag = True
            elif token == ')':
                flag = False

    statements = [(tokens[item[0]: item[1]+1], item) for item in intervals]
    return statements


def slicing_mask(front, back):
    tokens = back
    seeds = set()
    for i, token in enumerate(tokens):
        if is_identifier(token):
            if i < len(tokens) - 1 and tokens[i+1] != '(' and not is_identifier(tokens[i+1]):
                seeds.add(token)

    tokens = front
    statements = get_statements(tokens)

    st_list = []

    for n, st in enumerate(reversed(statements)):
        flag = False
        assignment_flag = False
        depend = False
        for i, token in enumerate(st[0]):
            if token is '=':
                flag = True

            if is_identifier(token) and not flag and token in seeds:
                depend = True
                assignment_flag = True
                continue
            if assignment_flag and flag:
                try:
                    if i < len(tokens) - 1 and is_identifier(token) and tokens[i+1] != '(':
                        seeds.add(token)
                except IndexError:
                    pass
        if depend:
            st_list.append(st[1])
    try:
        method_def = statements[0][1]
        if method_def not in st_list:
            st_list.append(method_def)
    except Exception as e:
        print("statements, error:", e)

    code = ' '.join(front)+' '+' '.join(back)
    mask = [0]*len(front)
    for item in st_list:
        mask[item[0]:item[1]] = [1]*(item[1]-item[0])
    print('sum(mask):', sum(mask))
    print('len(front):', len(front))
    print('len(mask):', len(mask))


    assert sum(mask) > 1 and len(front) == len(mask), print(code)

    return ' '.join(front), ' '.join(back), mask

def mask_slicing(dataset):
    origin_root = 'data/baseline/'
    with open(origin_root+'src-%s.txt'%dataset) as fps, open(origin_root+'tgt-%s.txt'%dataset) as fpt:
        origin_src = fps.readlines()
        origin_tgt = fpt.readlines()

    target_root = 'data/multi_slicing/'
    os.makedirs(target_root, exist_ok=True)
    with open(target_root+'src-nexgen-%s.front'%dataset, 'w') as fwf, open(target_root+'src-nexgen-%s.back'%dataset, 'w') as fwb,\
            open(target_root+'src-nexgen-%s.mask'%dataset, 'w') as fwm, \
            open(target_root+'tgt-nexgen-%s.txt'%dataset, 'w') as fwt:
        for i, (s, t) in enumerate(zip(origin_src, origin_tgt)):
            print(i)
            s = s.strip()
            if not re.match(r'\w+', s):
                print(s)
                s = re.sub(r'^.*?(\w+)', r' \1', s)
                print(s)
            s = re.sub(r'\\\\', ' ', s)
            s = re.sub(r'\\ "', ' \\"', s)
            try_idx = get_try_index(s)
            if not try_idx:
                print('try not found: ', s)
                exit(-1)
            s = s.split()
            front = s[:try_idx]
            back = s[try_idx:]
            front, back, mask = slicing_mask(front, back)
            mask = json.dumps(mask)
            fwf.write(front+'\n')
            fwb.write(back+'\n')
            fwm.write(mask+'\n')
            fwt.write(t)


def mask_slicing_with_ast(dataset):
    origin_root = 'data/baseline/'
    with open(origin_root+'src-%s.txt'%dataset) as fps, open(origin_root+'tgt-%s.txt'%dataset) as fpt:
        origin_src = fps.readlines()
        origin_tgt = fpt.readlines()

    target_root = 'data/multi_slicing/'
    os.makedirs(target_root, exist_ok=True)
    with open(target_root+'src-nexgen-ast-%s.front'%dataset, 'w', errors='ignore') as fwf, open(target_root+'src-nexgen-ast-%s.back'%dataset, 'w', errors='ignore') as fwb,\
            open(target_root+'tgt-nexgen-ast-%s.txt'%dataset, 'w', errors='ignore') as fwt, \
            open(target_root + 'src-nexgen-ast_tokens-%s.txt' % dataset, 'w' , errors='ignore') as fwtokens:
        # all_len = 0
        # len_256 = 0
        for i, (s, t) in enumerate(zip(origin_src, origin_tgt)):
            front, back = handle_codes_with_ast(s, t)

            fwf.write(" ".join(front)+'\n')
            fwb.write(" ".join(back)+'\n')
            fwtokens.write(s + '\n')
            fwt.write(t)


def handle_codes_with_ast(s, t = None):
    s = s.strip()
    if not re.match(r'\w+', s):
        print(s)
        s = re.sub(r'^.*?(\w+)', r' \1', s)
        print(s)
    s = re.sub(r'\\\\', ' ', s)
    s = re.sub(r'\\ "', ' \\"', s)

    func = s
    if (t != None):
        func = s + t + " }"
    func_tokens = ast_bert_data_process_util_for_catch.get_feature_tokens_for_catch(func)
    s = " ".join(func_tokens)
    try_idx = get_try_index_for_ast(func_tokens)
    catch_idex = get_catch_index_for_ast(func_tokens)
    if not try_idx:
        print('try not found: ', s)
        exit(-1)
    if not catch_idex:
        print('catch not found: ', s)
        exit(-1)

    front = func_tokens[:try_idx]
    back = func_tokens[try_idx:catch_idex]
    return front, back

def predict_data_process(code_lines, begin_position, end_position):
    code_lines = code_lines.split("\n")
    new_lines = '\n'.join(code_lines[0:begin_position]) + "\n"
    new_lines = new_lines + "try {" + "\n"
    new_lines = new_lines + '\n'.join(code_lines[begin_position: end_position + 1])  + '\n'
    new_lines = new_lines + '} catch ( Exception e ) { } ' + "\n"
    new_lines = new_lines + '\n'.join(code_lines[end_position + 1:])
    front, back = handle_codes_with_ast(s = new_lines)
    return ' '.join(front), ' '.join(back)


def use_mask_slicing(dataset, input_file):
    origin_root = 'data/baseline/'
    with open(input_file) as fps:
        origin_src = fps.readlines()

    target_root = 'data/multi_slicing/'
    os.makedirs(target_root, exist_ok=True)
    with open(target_root+'src-%s.front'%dataset, 'w') as fwf, open(target_root+'src-%s.back'%dataset, 'w') as fwb,\
            open(target_root+'src-%s.mask'%dataset, 'w') as fwm:
        for i, s in enumerate(origin_src):
            print(i)
            s = s.strip()
            if not re.match(r'\w+', s):
                print(s)
                s = re.sub(r'^.*?(\w+)', r' \1', s)
                print(s)
            s = re.sub(r'\\\\', ' ', s)
            s = re.sub(r'\\ "', ' \\"', s)
            try_idx = get_try_index(s)
            if try_idx < 0:
                print('try not found: ', s)
                exit(-1)
            s = s.split()
            front = s[:try_idx]
            back = s[try_idx:]
            front, back, mask = slicing_mask(front, back)
            mask = json.dumps(mask)
            fwf.write(front+'\n')
            fwb.write(back+'\n')
            fwm.write(mask+'\n')


def extract_first_exception_types(java_code):
    pattern = r'catch\s*\(\s*(?:final\s+)?([\w.|s]+)\s+([\w]+)\s*\)'
    match = re.search(pattern, java_code)
    if match:
        exceptions = [exception.strip() for exception in match.group(1).split('|')]
        return ', '.join(exceptions)
    return None

def extract_exception_to_file(dataset):
    origin_root = 'data/baseline/'
    print("extract_exception start")
    with open(origin_root + 'tgt-%s.txt' % dataset) as fpt:
        origin_tgt = fpt.readlines()
        with open(origin_root + 'excep-%s.txt' % dataset, 'w') as fwf:
            for i, t in enumerate(origin_tgt):
                exception = extract_first_exception_types(t)
                if(exception == None):
                    excep_res = 'Exception'
                    print("extract_exception res=exmpty, set default \"Exception\"")
                else:
                    excep_res = exception
                fwf.write(excep_res + '\n')
                print("extract_exception source:" + t + " \n exception: "+excep_res)
            print("extract_exception done")

def extract_exception(dataset):
    origin_root = 'data/multi_slicing/'
    print("extract_exception start")
    res = []
    with open(origin_root + 'tgt-nexgen-%s.txt' % dataset) as fpt:
        origin_tgt = fpt.readlines()
        for i, t in enumerate(origin_tgt):
            exception = extract_first_exception_types(t)
            if (exception == None):
                excep_res = 'Exception'
                print("extract_exception res=exmpty, set default \"Exception\"")
            else:
                excep_res = exception
            res.append(excep_res)
            print("extract_exception source:" + t + " \n exception: " + excep_res)
        print("extract_exception done")
    return res

def extract_exception_for_ast(dataset):
    origin_root = 'data/multi_slicing/'
    print("extract_exception start")
    res = []
    with open(origin_root + 'tgt-nexgen-ast-%s.txt' % dataset) as fpt:
        origin_tgt = fpt.readlines()
        for i, t in enumerate(origin_tgt):
            exception = extract_first_exception_types(t)
            if (exception == None):
                excep_res = 'Exception'
                print("extract_exception res=exmpty, set default \"Exception\"")
            else:
                excep_res = exception
            res.append(excep_res)
            print("extract_exception source:" + t + " \n exception: " + excep_res)
        print("extract_exception done")
    return res

def extract_exception_distinct(dataset):
    datas = extract_exception(dataset)
    res = set(datas)
    origin_root = 'data/baseline/'
    with open(origin_root + 'excep-distinct-%s.txt' % dataset, 'w') as fwf:
        for i in res:
            fwf.write(i+'\n')


def count_exception_types(exception_list):
    count_dict = {}
    for exception in exception_list:
        if exception in count_dict:
            count_dict[exception] += 1
        else:
            count_dict[exception] = 1
    sorted_result = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_result

def extract_exception_count(dataset):
    datas = extract_exception(dataset)
    count_exception = count_exception_types(datas)
    origin_root = 'data/baseline/'
    with open(origin_root + 'excep-count-%s.txt' % dataset, 'w') as fwf:
        for exception, count in count_exception:
            fwf.write(exception + ":" + str(count) + "\n")

def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                result_dict[key] = int(value)
    return result_dict

def read_txt_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines

def write_values_to_txt(dict_data, keys, output_file):
    with open(output_file, 'w') as file:
        for key in keys:

            value = '0'
            if (key in dict_data):
                value = dict_data[key]
            file.write(f"{value}\n")

def write_text_to_txt(txts, output_file):
    with open(output_file, 'w') as file:
        for txt in txts:
            file.write(f"{txt}\n")


def handle_exception_data(dataset):
    out_root = 'data/multi_slicing/'
    origin_root = 'data/baseline/'
    result_dict = read_txt_to_dict(origin_root + 'exception-class-map.txt')
    write_values_to_txt(result_dict, extract_exception(dataset), out_root + 'nexgen-%s-label.txt' % dataset)


def handle_exception_data_from_nexgen_ast(dataset):
    out_root = 'data/multi_slicing/'
    origin_root = 'data/baseline/'
    result_dict = read_txt_to_dict(origin_root + 'exception-class-map.txt')
    write_values_to_txt(result_dict, extract_exception_for_ast(dataset), out_root + 'nexgen-ast-%s-label.txt' % dataset)


def handle_exception_data_from_drex(dataset):
    out_root = 'data/drex/'
    origin_root = 'data/baseline/'
    try_in_all = []
    try_before_all = []
    exception_all = []

    with open(origin_root + '200k_%s.txt' % dataset) as fpt:
        origin_tgt = fpt.readlines()
        for i, t in enumerate(origin_tgt):
            lines = t.split("#")
            if(len(lines) < 4):
                continue
            method_input = lines[0]
            codes = lines[1]
            exception = lines[2].split(",")[0]
            try_index = lines[3]
            try_indexs = try_index.split(",")
            try_token_begin_index = int(try_indexs[0])
            try_token_end_index = int(try_indexs[1].replace("\n", ""))
            code_tokens = codes.split(",")
            method_input_split = method_input.replace(",", " ")
            try_before = method_input_split + " " + " ".join(code_tokens[0:try_token_begin_index])
            try_in = " ".join(code_tokens[try_token_begin_index : try_token_end_index + 1])
            try_before_all.append(try_before)
            try_in_all.append(try_in)
            exception_all.append(exception)


    write_text_to_txt(try_before_all, out_root + 'src-%s.front' % dataset )

    write_text_to_txt(try_in_all, out_root + 'src-%s.back' % dataset )


    result_dict = read_txt_to_dict(origin_root + 'exception-class-map-drex-52.txt')
    write_values_to_txt(result_dict, exception_all, out_root + '%s-label.txt' % dataset)


def get_exception_from_drex(dataset):
    origin_root = 'data/baseline/'
    exception_all = []

    with open(origin_root + '200k_%s.txt' % dataset) as fpt:
        origin_tgt = fpt.readlines()
        for i, t in enumerate(origin_tgt):
            lines = t.split("#")
            if(len(lines) < 4):
                continue
            exception = lines[2].split(",")[0]
            exception_all.append(exception)
    return exception_all

def extract_exception_count_drex(dataset):
    datas = get_exception_from_drex(dataset)
    count_exception = count_exception_types(datas)
    origin_root = 'data/drex/'
    with open(origin_root + 'excep-count-all-%s.txt' % dataset, 'w') as fwf:
        for i, (exception, count) in enumerate(count_exception):
            fwf.write(exception + ":" + str(count) + "\n")


if __name__ == '__main__':
    # extract_exception_count('train')
    # extract_exception('train')
    # extract_exception_distinct('train')


    extract_exception_count_drex('train')
    extract_exception_count_drex('test')
    extract_exception_count_drex('valid')
    handle_exception_data_from_drex('train')
    handle_exception_data_from_drex('test')
    handle_exception_data_from_drex('valid')


    # mask_slicing('train')
    # mask_slicing('valid')
    # mask_slicing('test')
    # handle_exception_data('test')
    # handle_exception_data('valid')
    # handle_exception_data('train')


    mask_slicing_with_ast('train')
    mask_slicing_with_ast('valid')
    mask_slicing_with_ast('test')
    handle_exception_data_from_nexgen_ast('test')
    handle_exception_data_from_nexgen_ast('valid')
    handle_exception_data_from_nexgen_ast('train')