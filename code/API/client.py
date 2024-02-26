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

# 获取向量库
# file_path=r"D:\work\QiJi\qiji_compet\code\data\dataset\multi_cls_data\train_multi_v2.xlsx"
# save_path=r"D:\work\QiJi\qiji_compet\code\ir\softmatch"
# url = fr"http://127.0.0.1:4567/soft_match/vector_update?file_path={file_path}&save_path={save_path}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)

# 获取索引库
# save_path=r"D:\work\QiJi\qiji_compet\code\ir\softmatch\vector.pkl"
# url = fr"http://127.0.0.1:4567/soft_match/index_update?vector_path={save_path}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)

# search
text = "垃圾"
# url = f"http://127.0.0.1:4567/soft_match/text2embedding?contents={text}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)

topk = 5
url = f"http://127.0.0.1:4567/soft_match/search?text={text}&topk={topk}"
r = requests.get(url=url)
print(r.text)
result_json = json.loads(r.text)
print(result_json)
