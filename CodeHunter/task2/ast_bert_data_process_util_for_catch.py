import logging
import re

import javalang

logging.basicConfig(filename='new_data_process.log', level=logging.DEBUG)

parse_error_count = 0


def get_feature_tokens_for_catch(code_lines, depth = 0):
    if(depth > 2):
        return code_lines.split()

    old_lines = code_lines

    code_lines = re.sub(r"\s*([().])\s*", r"\1", code_lines)
    code_lines = code_lines.replace("\r", " ")
    code_lines = code_lines.replace("\t", " ")
    feature_tokens = []
    new_lines = []
    try:
        # tree = javalang.parse.parse("public void test(){log.info("")};")
        # codestr = '\n'.join(code_lines)
        codestr = code_lines
        # features = extract_java_function_features(code_lines)
        programtokens = javalang.tokenizer.tokenize(codestr)
        # token_str = extract_statement_level_info(codestr)
        # print("programtokens",list(programtokens))
        parser = javalang.parse.Parser(programtokens)
        programast = parser.parse_member_declaration()

        feature_tokens = parse_feature_tokens(programast)
        final_feature_tokens  = []
        for token in feature_tokens:
            if len(token) > 2 :
                final_feature_tokens.extend(re.findall(r'\w+|[.]', token))
            else:
                final_feature_tokens.append(token)

        return final_feature_tokens
    except Exception as e:
        print(e)
        return get_feature_tokens_for_catch(code_lines + " }",depth + 1)


def extract_java_function_features(java_code):
    features = ""

    try:
        tree = javalang.parse.parse(java_code)
    except javalang.parser.JavaSyntaxError as e:
        return f"Error parsing Java code: {e.description}"

    for path, node in tree.filter(javalang.tree.MethodDeclaration):

        start_line, end_line = node.position[0], node.position[2]
        function_code = java_code.splitlines()[start_line - 1:end_line]


        features += '\n'.join(function_code) + '\n\n'

    return features


def parse_feature_tokens(programast):
    codes = []
    try:
        if (programast == None):
            return []

        if hasattr(programast, 'return_type'):
            if (isinstance(programast.return_type, list)):
                for i in programast.return_type:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.return_type))


        code_type = type(programast)
        code_type = str(code_type.__name__).split('.')[-1]

        if code_type == 'LocalVariableDeclaration':
            codes.append('VarDec')
        elif code_type == 'TryStatement':
            codes.append('TRY')
        elif code_type == 'MethodInvocation':
            codes.append('METHOD')
            codes.append('INVOCATION')
        elif code_type == 'ReturnStatement':
            codes.append('RETURN')
        elif (code_type != 'str' and code_type != '' and code_type != 'StatementExpression'
              and code_type != 'MemberReference' and code_type != 'ReferenceType'
              and code_type != 'FormalParameter' and code_type != 'BasicType' and code_type != 'VariableDeclarator' and code_type != 'BlockStatement'):
            codes.append(code_type)
        if (code_type == 'tuple' or code_type == 'list'):
            return codes

        if hasattr(programast, 'type'):
            if (isinstance(programast.type, list)):
                for i in programast.type:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.type))

        if hasattr(programast, 'types'):
            if (isinstance(programast.types, list)):
                for i in programast.types:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.types))


        if (type(programast) == str):
            ast_str = str(programast)
            for str_sub in ast_str.split(" "):
                str_sub = str_sub.replace(" ", "")
                str_sub = str_sub.replace("\"", "")
                if str_sub != '':
                    codes.append(str_sub)
            return codes

        if hasattr(programast, 'operandl'):
            if (isinstance(programast.operandl, list)):
                for i in programast.operandl:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.operandl))

        if hasattr(programast, 'qualifier') and programast.qualifier != None and programast.qualifier != '':
            qualifier = programast.qualifier
            codes.append('qualifier')
            codes.append(qualifier)
        if hasattr(programast, 'name') and programast.name != None and programast.name !='':
            name = programast.name
            codes.append(name)
        if hasattr(programast, 'member') and programast.member != None and programast.member !='':
            member = programast.member
            codes.append(member)
        if hasattr(programast, 'operator') and programast.operator != None and programast.operator !='':
            operator = programast.operator
            codes.append(operator)



        if hasattr(programast, 'sub_type'):
            if (isinstance(programast.sub_type, list)):
                for i in programast.sub_type:
                    if i != None:
                        codes.extend(".")
                    codes.extend(parse_feature_tokens(i))
            else:
                if programast.sub_type != None:
                    codes.extend(".")
                codes.extend(parse_feature_tokens(programast.sub_type))

        if (hasattr(programast, 'parameters') and programast.parameters != None):
            codes.append("parameters")
            if (isinstance(programast.parameters, list)):
                codes.append("(")
                for i in programast.parameters:
                    codes.extend(parse_feature_tokens(i))
                codes.append(")")
            else:
                codes.append("(")
                codes.extend(parse_feature_tokens(programast.parameters))
                codes.append(")")

        if (hasattr(programast, 'parameter') and programast.parameter != None):
            codes.append("parameter")
            if (isinstance(programast.parameter, list)):
                codes.append("(")
                for i in programast.parameter:
                    codes.extend(parse_feature_tokens(i))
                codes.append(")")
            else:
                codes.append("(")
                codes.extend(parse_feature_tokens(programast.parameter))
                codes.append(")")

        if (hasattr(programast, 'throws') and programast.throws != None):
            codes.append("throws")
            if (isinstance(programast.throws, list)):
                for i in programast.throws:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.throws))

        if (hasattr(programast, 'arguments') and programast.arguments != None):
            codes.append("arguments")
            if (isinstance(programast.arguments, list)):
                codes.append("(")
                for i in programast.arguments:
                    codes.extend(parse_feature_tokens(i))
                codes.append(")")
            else:
                codes.append("(")
                codes.extend(parse_feature_tokens(programast.arguments))
                codes.append(")")

        if (hasattr(programast, 'expressionl') and programast.expressionl != None):
            if (isinstance(programast.expressionl, list)):
                for i in programast.expressionl:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.expressionl))

        if (hasattr(programast, 'expressionr') and programast.expressionr != None):
            if (isinstance(programast.expressionr, list)):
                for i in programast.expressionr:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.expressionr))

        if (hasattr(programast, 'value') and programast.value != None):
            codes.append("value")
            if (isinstance(programast.value, list)):
                for i in programast.value:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.value))

        if hasattr(programast, 'operandr'):
            codes.append("operandr")
            if (isinstance(programast.operandr, list)):
                for i in programast.operandr:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.operandr))

        if hasattr(programast, 'selectors'):
            if (isinstance(programast.selectors, list)):
                for i in programast.selectors:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.selectors))

        if hasattr(programast, 'expression'):
            if (isinstance(programast.expression, list)):
                for i in programast.expression:
                    codes.extend(parse_feature_tokens(i))
                    codes.append(";")
            else:
                codes.extend(parse_feature_tokens(programast.expression))
                codes.append(";")

        if hasattr(programast, 'declarators'):
            if (isinstance(programast.declarators, list)):
                for i in programast.declarators:
                    codes.extend(parse_feature_tokens(i))
                    codes.append(";")
            else:
                codes.extend(parse_feature_tokens(programast.declarators))
                codes.append(";")

        if hasattr(programast, 'initializer'):
            codes.append("initializer")
            if (isinstance(programast.initializer, list)):
                for i in programast.initializer:
                    codes.extend(parse_feature_tokens(i))
                    codes.append(";")
            else:
                codes.extend(parse_feature_tokens(programast.initializer))
                codes.append(";")

        if (hasattr(programast, 'condition') and programast.condition != None):
            if (isinstance(programast.condition, list)):
                codes.append("(")
                for i in programast.condition:
                    codes.extend(parse_feature_tokens(i))
                codes.append(")")
            else:
                codes.append("(")
                codes.extend(parse_feature_tokens(programast.condition))
                codes.append(")")


        if (hasattr(programast, 'body') and programast.body != None):
            if(isinstance(programast.body, list)):
                codes.append("{")
                for i in programast.body:
                    codes.extend(parse_feature_tokens(i))
                codes.append("}")
            else:
                codes.append("{")
                codes.extend(parse_feature_tokens(programast.body))
                codes.append("}")

        if (hasattr(programast, 'statements') and programast.statements != None):
            if (isinstance(programast.statements, list)):
                for i in programast.statements:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.statements))


        if (hasattr(programast, 'then_statement') and programast.then_statement != None):
            if (isinstance(programast.then_statement, list)):
                for i in programast.then_statement:
                    codes.append("{")
                    codes.extend(parse_feature_tokens(i))
                    codes.append("}")
            else:
                codes.append("{")
                codes.extend(parse_feature_tokens(programast.then_statement))
                codes.append("}")

        if (hasattr(programast, 'else_statement') and programast.else_statement != None):
            codes.append("ELSE")
            if (isinstance(programast.else_statement, list)):
                for i in programast.else_statement:
                    codes.append("{")
                    codes.extend(parse_feature_tokens(i))
                    codes.append("}")
            else:
                codes.append("{")
                codes.extend(parse_feature_tokens(programast.else_statement))
                codes.append("}")
            # res.update(parse_feature_tokens(programast.else_statement))
            # for i in programast.else_statement:
            #     res.update(parse_feature_tokens(i))

        if (hasattr(programast, 'block') and programast.block != None):
            if (isinstance(programast.block, list)):
                for i in programast.block:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.block))


        if (hasattr(programast, 'catches') and programast.catches != None):
            if (isinstance(programast.catches, list)):
                for i in programast.catches:
                    codes.append("{")
                    codes.extend(parse_feature_tokens(i))
                    codes.append("}")

            else:
                codes.append("{")
                codes.extend(parse_feature_tokens(programast.catches))
                codes.append("}")



        if (hasattr(programast, 'finally_block') and programast.finally_block != None):
            if (isinstance(programast.finally_block, list)):
                for i in programast.finally_block:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.finally_block))


        if (hasattr(programast, 'cases') and programast.cases != None):
            if (isinstance(programast.cases, list)):
                for i in programast.cases:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.cases))

        if (hasattr(programast, 'prefix_operators') and programast.prefix_operators != None):
            if (isinstance(programast.prefix_operators, list)):
                for i in programast.prefix_operators:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.prefix_operators))

        if (hasattr(programast, 'invocation') and programast.invocation != None):
            codes.append("invocation")
            if (isinstance(programast.invocation, list)):
                for i in programast.invocation:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.invocation))

        if (hasattr(programast, 'identifier') and programast.identifier != None):
            codes.append("identifier")
            if (isinstance(programast.identifier, list)):
                for i in programast.identifier:
                    codes.extend(parse_feature_tokens(i))
            else:
                codes.extend(parse_feature_tokens(programast.identifier))
    except Exception as e:
        logging.error("parse_feature_tokens error", programast, e)
    return codes


def extract_statement_level_info(java_code):
    parser = javalang.parser.Parser(javalang.tokenizer.tokenize(java_code))
    member_declarations = parser.parse_member_declaration()

    statement_info = ""
    if isinstance(member_declarations, javalang.tree.MethodDeclaration):
        for statement in member_declarations.body:
            statement_info += str(statement) + '\n'

    return statement_info


if __name__ == '__main__':
    str = 'public Object logAround ( ProceedingJoinPoint joinPoint ) throws Throwable { if ( log . isDebugEnabled ( ) ) { log . debug (  " Enter: {}.{}() with argument[s] = {} "  , joinPoint . getSignature ( ) . getDeclaringTypeName ( ) , joinPoint . getSignature ( ) . getName ( ) , Arrays . toString ( joinPoint . getArgs ( ) ) ) ; } try { Object result = joinPoint . proceed ( ) ; if ( log . isDebugEnabled ( ) ) { log . debug (  " Exit: {}.{}() with result = {} "  , joinPoint . getSignature ( ) . getDeclaringTypeName ( ) , joinPoint . getSignature ( ) . getName ( ) , result ) ; } return result ; }catch ( IllegalArgumentException e ) { log . error (  " Illegal argument: {} in {}.{}() "  , Arrays . toString ( joinPoint . getArgs ( ) ) , joinPoint . getSignature ( ) . getDeclaringTypeName ( ) , joinPoint . getSignature ( ) . getName ( ) ) ; throw e ; } }'
    tokens = get_feature_tokens_for_catch(str)
    print(tokens)