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
url = r"http://127.0.0.1:4567/soft_match/search?text=狗东西&topk=5"
if __name__ == "__main__":
    r = requests.get(url=url)
    result_json = json.loads(r.text)
    print(result_json)
