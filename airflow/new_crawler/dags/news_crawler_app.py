import pymysql
import requests
import json
import time
from bs4 import BeautifulSoup
import datetime
from datetime import datetime as dt
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator


default_args = {
    'owner': 'Jimmysu',
    'start_date': dt(2022,4,8,0,0),
    'schedule_interval': '@daily',
    'retries': 2,
    'retry_delay': timedelta(minutes=1)
}

db_settings  = {
	"host": "localhost",
	"port": 3306,
	"user": "root",
	"password": "as098765",
	"db" : "news_database",
	"charset": "utf8"
}

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18362',
}

def news_id(**data):
	start = data['data_interval_start']
	end = data['data_interval_end']
	start_stamp = int(dt.timestamp(dt.strptime(str(start).split('T')[0], "%Y-%m-%d")))
	end_stamp = int(dt.timestamp(dt.strptime(str(end).split('T')[0], "%Y-%m-%d"))-1)
	url ='https://news.cnyes.com/api/v3/news/category/tw_stock?startAt={}&endAt={}&limit=30'.format(start_stamp,end_stamp)
	res = requests.get(url, headers)
	newsID_list=[]
	last_page = json.loads(res.text)['items']['last_page']

	print('總共 {} 頁'.format(last_page))

	newsIDlist=json.loads(res.text)['items']['data']
	for i in newsIDlist:
		newsID=i['newsId']
		newsID_list.append(newsID)
	time.sleep(1)

	for p in range(2, last_page+1):
		url ='https://news.cnyes.com/api/v3/news/category/tw_stock?startAt={}&endAt={}&limit=30&page={}'.format(start_stamp,end_stamp,p)
		res = requests.get(url, headers)
		last_page = json.loads(res.text)['items']['last_page']
		print('正在讀取第 {} 頁'.format(p))
		newsIDlist=json.loads(res.text)['items']['data']
		for i in newsIDlist:
			newsID=i['newsId']
			newsID_list.append(newsID)
		time.sleep(1)
	print("共獲取 {} 篇新聞：".format(len(newsID_list)))
	return newsID_list

def news_crawler(**data):
	news_id = data['task_instance'].xcom_pull(task_ids='get_news_id')
	print("抓取新聞")
	result = []
	for idx, i in enumerate(news_id):
		temp = {}
		url = "https://news.cnyes.com" + "/news/id/{}?exp=a".format(i)
		temp['url'] = url
		res =  requests.get(url)
		soup = BeautifulSoup(res.text,'html.parser')
		try:
			if len(list(soup.find('div',{'class':'_1UuP'}).text)) ==0:
				continue
			elif len(list(soup.find('h1').text)) ==0 :
				continue
			else:
				temp['Title'] = soup.find('h1').text
				Date = soup.find('time').text.split()
				temp['Date'] = Date[0].replace('/', '-')
				# print(soup.find('time').text)
				temp['Content'] = soup.find('div',{'class':'_1UuP'}).text
		except:
			continue
		result.append(temp)
		time.sleep(3)
	print("抓取完畢")
	return result

def process_data(mode, **data):

	try:

		count = 0
		connect = pymysql.connect(**db_settings)

		with connect.cursor() as cursor:
			if mode == 'write':
				print("將資料寫入資料庫")
				news_list = data['task_instance'].xcom_pull(task_ids='get_news_data')
				command = "INSERT INTO news(Title, Content, Date, url)VALUES(%s, %s, %s, %s)"
				for item in news_list:
					cursor.execute(command, (item['Title'], item['Content'], item['Date'], item['url']))
					count += 1
				print("成功寫入 {} 筆資料".format(count))
				connect.commit()
			elif mode == 'read':
				command = "SELECT * FROM news WHERE Date = {}".format("'" + str(data['ds']) + "'")
				cursor.execute(command)
				result = cursor.fetchall()
				print(result)

	except Exception as ex:
		print(ex)


with DAG('news_crawler_app', default_args=default_args) as dag:

	get_news_id = PythonOperator(
		task_id = 'get_news_id',
		python_callable = news_id
		)

	get_news_data = PythonOperator(
		task_id = 'get_news_data',
		python_callable = news_crawler,

		)

	do_write_data = PythonOperator(
		task_id = "do_write_data",
		python_callable = process_data,
		op_args = ['write']
		)

	do_read_data = PythonOperator(
		task_id = "do_read_data",
		python_callable = process_data,
		op_args = ['read']
		)
	do_nothing = DummyOperator(task_id = 'do_nothing')

	# workflow
	get_news_id >> get_news_data >> do_write_data 
	do_write_data >> do_read_data
