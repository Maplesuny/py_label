from fastapi import APIRouter
from app.core.eeg_data_processor import get_eeg_data, get_patient_info, get_data_sec
from fastapi.responses import JSONResponse
import os
import sys
import psycopg2 as pg2
import json
import datetime

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


@eeg_router.get('/gettablenames')
def getDBAlltables():
    save_tablenames = []
    conn, cur = getDBConnection()
    cur.execute("SELECT * from pg_tables where schemaname = 'public'")
    conn.commit()
    table_data = cur.fetchall()

    for item in table_data:
        save_tablenames.append(item[1])
    return save_tablenames


@eeg_router.get('/gettable_column/{table}')
def getpostgreSQL_columns(table):
    conn, cur = getDBConnection()
    cur.execute('SELECT * FROM public."' + table + '" limit 0')
    conn.commit()
    colnames = [desc[0] for desc in cur.description]
    return colnames


@eeg_router.get('/gettable_data/{table}')
def gettable_data(table):
    col_list = getpostgreSQL_columns(table)
    conn, cur = getDBConnection()
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

# ---------------------------------------------------------------------


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
