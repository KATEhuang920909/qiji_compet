# -*- coding =  utf-8 -*-
"""
@Time : 2024/3/4 9:26
@Author: huangkai2
@File:config.py
@Software: PyCharm
"""
import requests
import json


# search
text ='你脑子里是有通古斯爆炸留下de坑？'
topk = 10
url = f"http://127.0.0.1:4567/soft_match/search?text={text}&topk={topk}"
r = requests.get(url=url)
result_json = json.loads(r.text)
print(result_json)