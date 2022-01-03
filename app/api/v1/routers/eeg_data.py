from fastapi import APIRouter
from numpy import array
from app.core.eeg_data_processor import get_eeg_data, get_patient_info, get_data_sec
from app.core.Seizure_detect_pipeline import return_pred_result
from fastapi.responses import JSONResponse
import os
import sys
import psycopg2 as pg2
import json
import datetime
import time
import random

eeg_router = APIRouter()


def getDBConnection():
    connection = pg2.connect(
        database="CUMH_TSR",
        user='readonly',
        password='readonly',
        host='10.23.215.31',
        port='15432'
    )
    cur = connection.cursor()
    return connection, cur


def getDBConnection_eeg():
    connection = pg2.connect(
        database="testdb",
        user='postgres',
        password='god3214',
        host='10.65.51.164',
        port='5432'
    )
    cur = connection.cursor()
    return connection, cur

# def getDBConnection_eeg():
#     connection = pg2.connect(
#         database="testdb",
#         user='postgres',
#         password='1234',
#         host='192.168.0.13',
#         port='5432'
#     )
#     cur = connection.cursor()
#     return connection, cur


def restart_id():
    reconn, recur = getDBConnection_eeg()
    recur.execute('alter table eeg drop column id;')
    recur.execute('ALTER TABLE eeg ADD COLUMN id serial;')
    reconn.commit()
    print('重置id完成')


@eeg_router.get('/gettablenames')
def getDBAlltables(database: str):
    save_tablenames = []
    if(database == 'eeg'):
        conn, cur = getDBConnection_eeg()
    else:
        conn, cur = getDBConnection()
    cur.execute("SELECT * from pg_tables where schemaname = 'public'")
    conn.commit()
    table_data = cur.fetchall()

    for item in table_data:
        save_tablenames.append(item[1])
    return save_tablenames


@eeg_router.get('/gettable_column/{table}')
def getpostgreSQL_columns(table, database: str):
    if(database == 'eeg'):
        conn, cur = getDBConnection_eeg()
    else:
        conn, cur = getDBConnection()
    cur.execute('SELECT * FROM public."' + table + '" limit 0')
    conn.commit()
    colnames = [desc[0] for desc in cur.description]
    return colnames


@eeg_router.get('/gettable_data/{table}')
def gettable_data(table, database: str):
    col_list = getpostgreSQL_columns(table, database)
    if(database == 'eeg'):
        conn, cur = getDBConnection_eeg()
    else:
        conn, cur = getDBConnection()
    # conn, cur = getDBConnection()
    # cur.execute('SELECT * FROM public."' + table + '" limit 100000')
    cur.execute('SELECT * FROM public."' + table + '"')
    # print('SELECT * FROM public."' + table + '"')
    conn.commit()
    data = cur.fetchall()
    dict_data = {}
    save_data_list = []
    for item in data:
        for i in col_list:
            value = item[col_list.index(i)]
            datatime_check = isinstance(value, datetime.date)
            if datatime_check == True:
                dict_data[i] = value.strftime('%Y-%m-%d %H:%M:%S')
            else:
                dict_data[i] = item[col_list.index(i)]
        dicp_copy = dict_data.copy()
        save_data_list.append(dicp_copy)
    return save_data_list


# range0 = [1000, 200]
# range1 = [2000, 300]
# current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

@eeg_router.get('/get_eeg_data')
def select_data():
    conn, cur = getDBConnection_eeg()
    select_url = 'select count(*) from eeg '
    cur.execute(select_url)
    conn.commit()
    data = cur.fetchone()
    for item in data:
        # print(item)
        return item


@eeg_router.post('/insert_data')
def insert_data_eeg(page: int, starttime: float, endtime: float, range0: str, range1: str):
    # range0 = [1000, 2000]
    # range1 = [2000, 300]
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn, cur = getDBConnection_eeg()
    sql_comman = 'INSERT INTO eeg (page,starttime,endtime,range0,range1,systemtime) VALUES' + '(' + str(page) + ',' + str(starttime) + ',' + str(
        endtime) + ','+'array' + range0 + ',array' + range1+',' + 'TIMESTAMP '+'\'' + current_time + '\''+')' + ';'
    cur.execute(sql_comman)
    conn.commit()
    print('Done save')


@eeg_router.post('/delete_data')
def delete_data():
    conn, cur = getDBConnection_eeg()
    # delete_sql = 'delete from eeg where id IN (select id from eeg order by id DESC limit 1)'
    # 刪除前一筆
    delete_sql = 'delete from eeg where id IN (select min(id) from eeg where id in (select id from eeg order by id desc limit 2))'
    data_length = select_data()
    print(data_length)
    if data_length <= 1:
        return
    else:
        cur.execute(delete_sql)
        conn.commit()
        restart_id()
    print('刪除單筆完成')

# ---------------------------------------------------------------------

# 預測模型


@eeg_router.get('/pre_result')
def result():
    pre_result = return_pred_result()
    return pre_result


@eeg_router.get('/eegData')
def get_egg_data(start_time: int, end_time: int, montage_type: int):
    print('test')
    result_list = get_eeg_data(start_time, end_time, montage_type)
    if result_list:
        return result_list
    else:
        return JSONResponse(
            status_code=404,
            content={"message": f"Index not found"},
        )


@eeg_router.get('/test')
def patient_id():
    print('test')
    return 'test'


@eeg_router.get('/Time_Information')
def data_sec():
    sec = get_data_sec()
    return sec


@eeg_router.get('/eegPatientInfo/{patient_id}')
def eeg_patient_info(patient_id: str):
    print('id: ', patient_id)

    return get_patient_info(patient_id)


@eeg_router.get('/eegDataQuery/{id}')
def get_eeg_data2(id: str, start_time: int = 0, end_time: int = 0):
    print('id: ', id)
    print('start_time: ', start_time)
    print('end_time: ', end_time)

    return get_eeg_data(start_time, end_time)
