import os
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from selenium import webdriver
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.latest_only import LatestOnlyOperator

default_args = {
    # owner 
    'owner': 'Jimmy Su',
    # 開始日期
    'start_date': datetime(2022, 4, 3, 0, 0),
    # 排成日期：每日
    'schedule_interval': '@daily',
    # 失敗會重試兩次
    'retries': 2,
    # 重試間隔時間
    'retry_delay': timedelta(minutes=1)
}

# 爬蟲網址(漫畫人)
comic_page_template = 'https://www.manhuaren.com/manhua-{}'


def process_metadata(mode, **context):
    file_dir = os.path.dirname(__file__)
    metadata_path = os.path.join(file_dir, '../data/comic_data.json')
    if mode == 'read':
        with open(metadata_path, 'r') as fp:
            metadata = json.load(fp)
            print("Read History loaded: {}".format(metadata))
            return metadata
    elif mode == 'write':
        print("Saving latest comic information..")
        _, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')

        # update to latest chapter
        for comic_id, comic_info in dict(all_comic_info).items():
            all_comic_info[comic_id]['previous_chapter_num'] = comic_info['latest_chapter_num']

        with open(metadata_path, 'w') as fp:
            json.dump(all_comic_info, fp, indent=2, ensure_ascii=False)


def check_comic_info(**context):
    metadata = context['task_instance'].xcom_pull(task_ids='get_read_history')
    driver = webdriver.Chrome()
    driver.get('https://www.manhuaren.com/')

    all_comic_info = metadata
    anything_new = False
    for comic_id, comic_info in dict(all_comic_info).items():
        comic_name = comic_info['name']
        comic_url = comic_info['url']
        print("Fetching {}'s chapter list..".format(comic_name))
        driver.get(comic_page_template.format(comic_url))

        # get the latest chapter number
        links = driver.find_elements_by_partial_link_text('第')

        num = [s for s in links[0].text if s.isdigit()]
        latest_chapter_num = ""
        for i in num:
            latest_chapter_num += i
        latest_chapter_num = int(latest_chapter_num)
        previous_chapter_num = comic_info['previous_chapter_num']
        all_comic_info[comic_id]['latest_chapter_num'] = latest_chapter_num
        all_comic_info[comic_id]['new_chapter_available'] = latest_chapter_num > previous_chapter_num
        if all_comic_info[comic_id]['new_chapter_available']:
            anything_new = True
            print("There are new chapter for {}(latest: {})".format(comic_name, latest_chapter_num))

    if not anything_new:
        print("Nothing new now, prepare to end the workflow.")

    driver.quit()

    return anything_new, all_comic_info


def decide_what_to_do(**context):
    anything_new, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')

    print("跟紀錄比較，有沒有新連載？")
    if anything_new:
        return 'yes_generate_notification'
    else:
        return 'no_do_nothing'


def generate_message(**context):
    _, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')

    message = ''
    for comic_id, comic_info in all_comic_info.items():
        if comic_info['new_chapter_available']:
            name = comic_info['name']
            latest = comic_info['latest_chapter_num']
            prev = comic_info['previous_chapter_num']
            url = comic_info['url']
            message += '{} 最新一話： {} 話（上次讀到：{} 話）\n'.format(name, latest, prev)
            message += comic_page_template.format(url) + '\n\n'

    file_dir = os.path.dirname(__file__)
    message_path = os.path.join(file_dir, '../data/notification.txt')
    with open(message_path, 'w') as fp:
        fp.write(message)

def send_line_notify(**context):
    # 串連line api
    url = 'https://notify-api.line.me/api/notify'
    token = '81A77DbhBr6EuCHA7uftWjlXPKb6rYDQqTUXgIpYTZz'
    headers = {
        'content-type':
        'application/x-www-form-urlencoded',
        'Authorization': 'Bearer '+token
    }

    file_dir = os.path.dirname(__file__)
    message_path = os.path.join(file_dir, '../data/message.txt')
    with open(message_path, 'r') as fp:
        message = fp.read()

    # 用line bot 傳送訊息
    r = requests.post(url, headers=headers, data={'message': message})
    print("產生要寄給 Line 的訊息內容並存成檔案")


with DAG('comic_app', default_args=default_args) as dag:

    # define tasks
    latest_only = LatestOnlyOperator(task_id='latest_only')

    get_read_history = PythonOperator(
        task_id='get_read_history',
        python_callable=process_metadata,
        op_args=['read'],
    )

    check_comic_info = PythonOperator(
        task_id='check_comic_info',
        python_callable=check_comic_info,
    )

    decide_what_to_do = BranchPythonOperator(
        task_id='new_comic_available',
        python_callable=decide_what_to_do,
    )

    update_read_history = PythonOperator(
        task_id='update_read_history',
        python_callable=process_metadata,
        op_args=['write'],
    )

    generate_notification = PythonOperator(
        task_id='yes_generate_notification',
        python_callable=generate_message,
    )
    send_line_notification = PythonOperator(
        task_id='send_line_notify',
        python_callable=send_line_notify
    )


    do_nothing = DummyOperator(task_id='no_do_nothing')

    # define workflow
    latest_only >> get_read_history
    get_read_history >> check_comic_info >> decide_what_to_do
    decide_what_to_do >> generate_notification
    decide_what_to_do >> do_nothing
    generate_notification >> send_line_notification >> update_read_history
