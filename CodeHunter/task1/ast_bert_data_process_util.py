import logging
import re

import javalang

logging.basicConfig(filename='new_data_process.log', level=logging.DEBUG)

parse_error_count = 0

PAD = 'PAD'

def get_feature_tokens(code_lines, tags):
    old_lines = code_lines
    code_lines = [re.sub(r"\s*([().])\s*", r"\1", line) for line in code_lines]
    code_lines = [line.replace("\r", " ") for line in code_lines]
    code_lines = [line.replace("\t", " ") for line in code_lines]


    feature_tokens = []
    new_tags  = []
    new_lines = []
    try:
        # tree = javalang.parse.parse("public void test(){log.info("")};")
        codestr = '\n'.join(code_lines)
        programtokens = javalang.tokenizer.tokenize(codestr)
        # print("programtokens",list(programtokens))
        parser = javalang.parse.Parser(programtokens)
        programast = parser.parse_member_declaration()

        feature_tokens = parse_feature_tokens(programast)

        feature_tokens.update({len(code_lines): ['MethodDeclaration', 'END']})
        sorted_keys = sorted(feature_tokens.keys())
        for i in range(len(sorted_keys)):

            line_number = sorted_keys[i]
            line_number_end = line_number + 1
            if(i < len(sorted_keys) - 1):
                line_number_end = sorted_keys[i + 1]
            new_line = ''

            for line_no in range(line_number - 1, line_number_end - 1):
                new_line = new_line + old_lines[line_no]
            new_line = new_line + ' '.join(feature_tokens.get(line_number))
            new_tags.append(tags[line_number - 1])
            new_lines.append(new_line)
        return new_lines,new_tags
    except Exception as e:
        # logging.error("get_feature_tokens error")
        pass
    return new_lines,new_tags


def get_feature_tokens_for_api(code_lines, tags):
    old_lines = code_lines

    code_lines = [re.sub(r"\s*([().])\s*", r"\1", line) for line in code_lines]
    code_lines = [line.replace("\r", " ") for line in code_lines]
    code_lines = [line.replace("\t", " ") for line in code_lines]


    feature_tokens = []
    new_tags  = []
    new_lines = []
    try:
        # tree = javalang.parse.parse("public void test(){log.info("")};")
        codestr = '\n'.join(code_lines)
        programtokens = javalang.tokenizer.tokenize(codestr)
        # print("programtokens",list(programtokens))
        parser = javalang.parse.Parser(programtokens)
        programast = parser.parse_member_declaration()

        feature_tokens = parse_feature_tokens(programast)

        feature_tokens.update({len(code_lines): ['MethodDeclaration', 'END']})
        for i in range(len(code_lines)):
            new_line = ''
            new_line = new_line + old_lines[i]
            if(feature_tokens.get(i + 1) != None):
                feature_token = ' '.join(feature_tokens.get(i + 1))
                new_line = new_line + feature_token
            new_tags.append(tags[i])
            new_lines.append(new_line)
        return new_lines,new_tags
    except Exception as e:
        # logging.error("get_feature_tokens error")
        pass
    return new_lines,new_tags


def parse_feature_tokens(programast):
    res = {}
    try:
        if (programast == None):
            return res

        code_type = type(programast)
        code_type = str(code_type.__name__).split('.')[-1]

        if (code_type == 'tuple' or code_type == 'list'):
            # for p in programast:
            #     res.update(parse_feature_tokens(p))
            return res

        now = []
        if (isinstance(programast, javalang.tree.MethodDeclaration)):
            now.append('START')

        if (programast.position != None):

            line_number = programast.position[0]
            now = [code_type]

            now = {line_number: now}
            res.update(now)

        # modifiers = []
        # if hasattr(programast, 'modifiers'):
        #     modifiers = [m for m in programast.modifiers]
        #     now.extend(modifiers)


        if (hasattr(programast, 'body') and programast.body != None):
            if(isinstance(programast.body, list)):
                for i in programast.body:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.body))

        if (hasattr(programast, 'statements') and programast.statements != None):
            if (isinstance(programast.statements, list)):
                for i in programast.statements:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.statements))

        if (hasattr(programast, 'then_statement') and programast.then_statement != None):
            if (isinstance(programast.then_statement, list)):
                for i in programast.then_statement:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.then_statement))
            # for i in programast.then_statement:
            #     res.update(parse_feature_tokens(i))

        if (hasattr(programast, 'else_statement') and programast.else_statement != None):
            if (isinstance(programast.else_statement, list)):
                for i in programast.else_statement:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.else_statement))
            # res.update(parse_feature_tokens(programast.else_statement))
            # for i in programast.else_statement:
            #     res.update(parse_feature_tokens(i))

        if (hasattr(programast, 'block') and programast.block != None):
            if (isinstance(programast.block, list)):
                for i in programast.block:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.block))

        if (hasattr(programast, 'catches') and programast.catches != None):
            if (isinstance(programast.catches, list)):
                for i in programast.catches:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.catches))


        if (hasattr(programast, 'finally_block') and programast.finally_block != None):
            if (isinstance(programast.finally_block, list)):
                for i in programast.finally_block:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.finally_block))


        if (hasattr(programast, 'cases') and programast.cases != None):
            if (isinstance(programast.cases, list)):
                for i in programast.cases:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.cases))

        if (hasattr(programast, 'expression') and programast.expression != None):
            if (isinstance(programast.expression, list)):
                for i in programast.expression:
                    res.update(parse_feature_tokens(i))
            else:
                res.update(parse_feature_tokens(programast.expression))
    except Exception as e:
        logging.error("parse_feature_tokens error", programast, e)
    return res