from fastapi import APIRouter
from numpy import array
from app.core.eeg_data_processor import get_eeg_data, get_patient_info, get_data_sec
from app.core.Seizure_detect_pipeline import return_pred_result
from fastapi.responses import JSONResponse
import os
import sys
import psycopg2 as pg2
import pandas as pd
import json
import datetime
import time
import random
from configparser import ConfigParser

eeg_router = APIRouter()


# 初始化設定
def config_setting(filename, section='postgresql'):
    # 初始化設定連線
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename + '.ini')
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename + '.ini'))
    return db


# 連接資料庫----看是要連接eeg還是cmuh
def connection(init):
    conn = None
    params = config_setting(init)
    conn = pg2.connect(**params)
    cur = conn.cursor()

    # DataFrame = pd.read_sql(query,conn)
    return conn, cur


def restart_id():
    reconn, recur = connection('eeg')
    recur.execute('alter table eeg drop column id;')
    recur.execute('ALTER TABLE eeg ADD COLUMN id serial PRIMARY KEY;')
    reconn.commit()
    print('重置id完成')


@eeg_router.post('/query')
def query_db():
    reconn, recur = connection('eeg')
    recur.execute('select * from eeg')
    reconn.commit()
    print('query 完成')


@eeg_router.get('/gettablenames')
def getDBAlltables(database: str):
    save_tablenames = []
    if(database == 'eeg'):
        conn, cur = connection('eeg')
    else:
        conn, cur = connection('cmuh')
    cur.execute("SELECT * from pg_tables where schemaname = 'public'")
    conn.commit()
    table_data = cur.fetchall()

    for item in table_data:
        save_tablenames.append(item[1])
    return save_tablenames


@eeg_router.get('/gettable_column/{table}')
def getpostgreSQL_columns(table, database: str):
    if(database == 'eeg'):
        conn, cur = connection('eeg')
    else:
        conn, cur = connection('cmuh')
    cur.execute('SELECT * FROM public."' + table + '" limit 0')
    conn.commit()
    colnames = [desc[0] for desc in cur.description]
    return colnames


@eeg_router.get('/gettable_data/{table}')
def gettable_data(table, database: str):
    col_list = getpostgreSQL_columns(table, database)
    if(database == 'eeg'):
        conn, cur = connection('eeg')
    else:
        conn, cur = connection('cmuh')
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


@eeg_router.get('/get_eeg_data')
def select_data():
    conn, cur = connection('eeg')
    select_url = 'select count(*) from eeg '
    cur.execute(select_url)
    conn.commit()
    data = cur.fetchone()
    for item in data:
        # print(item)
        return item


# 擷取目前頁數(需要傳入頁數)
@eeg_router.get('/page')
def getpage(page: int):
    print('後端收到目前頁數:', page)
    return page


# 測試


@eeg_router.get('/testd')
def testd(current_page):
    class Reading:
        def __init__(self, myid, done, title, events, page, start, end, channel, montage, features):
            self.id = myid
            self.done = done
            self.title = title
            self.events = events
            self.page = page
            self.starttime = start
            self.endtime = end
            self.channel = channel
            self.montage_type = montage
            self.features = features

    query = 'select id,title,done,events,page,starttime,endtime,channel,montage_type,features from eeg  where page=' + \
        current_page+' order by id'
    print('query', query)
    conn, cur = connection('eeg')
    cur.execute(query)
    conn.commit()
    result = pd.read_sql(query, conn)
    listOfReading = [(Reading(row.id, row.done, row.title, row.events, row.page, row.starttime, row.endtime, row.channel,
                              row.montage_type, row.features)) for index, row in result.iterrows()]
    return listOfReading


@eeg_router.get('/ee')
def testd():
    class Reading:
        def __init__(self, employee_id):
            self.Employee_id = employee_id

    query = 'select * from test'
    print('query', query)
    conn, cur = connection('eeg')
    cur.execute(query)
    conn.commit()
    result = pd.read_sql(query, conn)
    listOfReading = [(Reading(row.employee_id))
                     for index, row in result.iterrows()]
    return listOfReading


@eeg_router.post('/post1')
def cancelid():
    conn, cur = connection('eeg')
    cancel_sql = "insert into test (Employee_id) values ('T12345')"
    cur.execute(cancel_sql)
    conn.commit()


@eeg_router.get('/testddd')
def testddd(current_page):
    itemStyle = {'borderWidth': 3,
                 'color': "rgba(245,39,56,0)", 'borderColor': 'rgba(220,20,57,0.8)'}

    class Reading:
        def __init__(self, myid, done, title, events, page, start, end, channel, montage, features):
            self.id = myid
            self.done = done
            self.title = title
            self.events = events
            self.page = page
            self.starttime = start
            self.endtime = end
            self.channel = channel
            self.montage_type = montage
            self.features = features
            self.itemStyle = itemStyle

    query = 'select id,title,done,events,page,starttime,endtime,channel,montage_type,features from eeg  where page=' + \
        str(current_page)+' order by id'

    conn, cur = connection('eeg')
    cur.execute(query)
    conn.commit()
    result = pd.read_sql(query, conn)
    for index, row in result.iterrows():
        print(index, row)
    # listOfReading = [(Reading(row.id, row.done, row.title, row.events, row.page, row.starttime, row.endtime, row.channel,
    #                           row.montage_type, row.features)) for index, row in result.iterrows()]

    # return listOfReading
    listOfReading = []
    for index, row in result.iterrows():
        listOfReading.append([Reading(row.id, row.done, row.title, row.events, row.page,
                             row.starttime, row.endtime, row.channel, row.montage_type, row.features), Reading(row.id, row.done, row.title, row.events, row.page,
                             row.starttime, row.endtime, row.channel, row.montage_type, row.features)])

    return listOfReading


@eeg_router.get('/testd_all')
def testd():
    class Reading:
        def __init__(self, myid, done, title, events, start, end, channel, montage, features):
            self.id = myid
            self.done = done
            self.title = title
            self.events = events
            self.starttime = start
            self.endtime = end
            self.channel = channel
            self.montage_type = montage
            self.features = features

    query = 'select id,title,done,events,starttime,endtime,channel,montage_type,features from eeg  order by id'
    print('query', query)
    conn, cur = connection('eeg')
    cur.execute(query)
    conn.commit()
    result = pd.read_sql(query, conn)
    listOfReading = [(Reading(row.id, row.done, row.title, row.events, row.starttime, row.endtime, row.channel,
                              row.montage_type, row.features)) for index, row in result.iterrows()]
    return listOfReading


# get 座標
@eeg_router.get('/get_position')
def position(page: int):
    class Reading:
        def __init__(self, merge2):
            self.range = merge2
    select_sql = 'select range0|| array [range1] as range from eeg where page =' + \
        str(page)+' order by systemtime ASC'
    conn, cur = connection('eeg')
    cur.execute(select_sql)
    conn.commit()
    result = pd.read_sql(select_sql, conn)
    print('座標長度', len(result))
    listOfReading = [(Reading(row.range))
                     for index, row in result.iterrows()]
    return listOfReading


# 要刪除最後一筆之前，先取得最後一筆的座標
@eeg_router.get('/before_del_data')
def del_before_data(page: int, montage: str):
    conn, cur = connection('eeg')
    query_url = 'select range0,range1 from eeg where page=\'' + \
        str(page)+'\' and montage_type=\'' + \
        montage+'\' order by id desc limit 1'
    cur.execute(query_url)
    conn.commit
    data = cur.fetchone()
    return data


# update normal等等資訊
@eeg_router.post('/update_db')
def update_data(column: str, value: str, id: int):
    conn, cur = connection('eeg')
    print('value_update', value)
    new_value = value.split(',')
    # new_value = list(map(str, value.split(', ')))
    if column != 'channel':
        update_sql = 'update eeg set '+column + \
            ' =\'' + value + '\' where id = '+str(id)

    if column == 'channel':
        update_sql = 'update eeg set '+column + \
            ' =' + 'array'+str(new_value) + ' where id = '+str(id)

    cur.execute(update_sql)
    conn.commit()
    query_db()
    print('update_sql', update_sql)
    print('[Done] Update id: '+str(id) + ' 欄位: '+column + 'value: '+value)
    return testd()


@eeg_router.post('/insert_data')
def insert_data_eeg(page: int, starttime: float, endtime: float, range0: str, range1: str, montage_type: str, title: str):
    # range0 = [1000, 2000]
    # range1 = [2000, 300]
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn, cur = connection('eeg')
    sql_comman = 'INSERT INTO eeg (page,starttime,endtime,range0,range1,montage_type,title,done,systemtime) VALUES' + '(' + str(page) + ',' + str(starttime) + ',' + str(
        endtime) + ',array' + range0 + ',array' + range1 + ',\''+montage_type+'\'' + ',\''+title+'\'' + ',False'+',TIMESTAMP '+'\'' + current_time + '\''+')' + ';'
    print('ppppppppp', sql_comman)
    cur.execute(sql_comman)
    conn.commit()
    print('Done save')


# 給todo cancel的
@eeg_router.post('/Cancelid')
def cancelid(id: int):
    conn, cur = connection('eeg')
    cancel_sql = 'delete from eeg where id=' + str(id)
    cur.execute(cancel_sql)
    conn.commit()
    restart_id()
    print('[Done] 刪除id為: '+str(id) + ' 的資料')


# 刪除目前頁面，最後框選對象
@eeg_router.post('/delete_page_draw')
def delete_draw(page: int, montage: str):
    conn, cur = connection('eeg')
    # range0, range1 的座標
    position = del_before_data(page, montage)
    delete_sql = 'delete from eeg where id IN (select id from eeg where page=\''+str(
        page)+'\' and montage_type=\''+montage+'\' order by id desc limit 1)'
    cur.execute(delete_sql)
    conn.commit()
    restart_id()
    print('[Done] 刪除 montage_type= ' + montage +
          ', page 等於' + str(page) + ' 的最後一筆')
    return position
    # ---------------------------------------------------------------------


# 預測模型
@ eeg_router.get('/pre_result')
def result():
    pre_result = return_pred_result()
    print('---------------------------------')
    print('pre_result', pre_result)
    # 建立Title
    result_title = ['Model_result']
    result = pd.DataFrame(columns=result_title, data=pre_result)
    result.to_csv('Pre_report.csv', index=False, encoding="utf_8_sig")
    print('SAVE完成')
    return pre_result


@ eeg_router.get('/eegData')
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


@eeg_router.get('/Time_Information')
def data_sec():
    sec = get_data_sec()
    return sec
