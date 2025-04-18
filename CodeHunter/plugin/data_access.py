import pprint
import pymysql
from pymysql import cursors


def conn_mysql():
    return pymysql.connect(
        host="localhost",
        port= 3306,
        user= "root",
        password= 'taoqiping',
        database= "code_hunter",
        charset= "utf8"
    )




def query_data(sql):
    conn = conn_mysql()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        conn.close()




def insert_or_update_data(sql):
    conn = conn_mysql()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
    finally:
        conn.close()


def insert_code_hunter_record(code_data, line_predict, exception_predict, user_evaluation, line_artificial, exception_artificial):
    sql = "insert code_hunter_record (code_data,line_predict,exception_predict,user_evaluation,line_artificial,exception_artificial) " \
          "values ('{}','{}','{}',{:d},'{}','{}')".format(code_data, line_predict, exception_predict, user_evaluation, line_artificial, exception_artificial)
    insert_or_update_data(sql)


if __name__ == "__main__":
    sql = "select * from code_hunter_record"
    datas = query_data(sql)
    pprint.pprint(datas)
