# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     client
   Author :       huangkai
   date：          2024/2/24
-------------------------------------------------
"""
import json

import requests

# url = r"http://127.0.0.1:4567/soft_match/index_update?vector_path=D:\code\qiji_compet\code\models\vector.pkl"
# url = "http://127.0.0.1:4567/soft_match/vector_update?file_path=D:\code\qiji_compet\code\data\dataset\multi_cls_data\dev_multi_v2.xlsx"
# headers = {"Content-Type": "application/json"}
# url = r"http://127.0.0.1:4567/soft_match/index_update?vector_path=狗东西&topk=5"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)

# # 获取向量库
# file_path=r"D:\work\QIJI\qiji_compet\code\data\dataset\multi_cls_data\train_multi_v2.xlsx"
# save_path=r"D:\work\QIJI\qiji_compet\code\ir\softmatch"
# url = fr"http://127.0.0.1:4567/soft_match/vector_update?file_path={file_path}&save_path={save_path}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)
# #
# # 获取索引库
# save_path=r"D:\work\QIJI\qiji_compet\code\ir\softmatch\vector.pkl"
# url = fr"http://127.0.0.1:4567/soft_match/index_update?vector_path={save_path}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)

# search
# text ='虽然天气还不错，但是你这种行为让人不齿'
# embedding_type="pool"
# url = f"http://127.0.0.1:4567/soft_match/text2embedding?contents={text}&embedding_type={embedding_type}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json["embedding_result"])



text ='虽然天气还不错，但是你这种行为啥也不是'
topk = 10
url = f"http://127.0.0.1:4567/soft_match/search?text={text}&topk={topk}"
r = requests.get(url=url)
print(r.text)
result_json = json.loads(r.text)
print(result_json)

# text ="你是谁qwertyuiop"
# text ="打倒中共共产党，打倒中共,这个法轮功万岁，妈卖批也。。。。"
# # embedding_type="sequence"
# url = f"http://127.0.0.1:4567/hard_match/filter?contents={text}"
# r = requests.get(url=url)
# print(r.text)
# result_json = json.loads(r.text)
# print(result_json)


# text ="打倒中共共产党，打倒中共,这个法轮功万岁，妈卖批也。。。。"
# # embedding_type="sequence"
# url = f"http://127.0.0.1:4567/ner/person_info_check?contents={text}"
# r = requests.get(url=url)
# print(r.text)
# result_json = json.loads(r.text)
# print(result_json)