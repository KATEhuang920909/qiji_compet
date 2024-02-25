# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     KeyBert
   Author :       huangkai
   date：          2024/2/25
-------------------------------------------------
"""
import requests
import jieba
import json
import numpy as np
from keybert import KeyBERT
url = r"http://127.0.0.1:4567/soft_match/text2embedding"
text_bag = jieba.lcut("你这种行为是死妈行为")
print(text_bag)
print(type(text_bag))
text_bag=["你","这种","行为","是","死","妈","行为"]
data = {"contents": text_bag}  # 需要传递的列表信息
data = json.dumps(data)
response = requests.post(url, data=data)
result = response.text
result = eval(result)["embedding_result"]
# print(result)
data = {"contents": ["尼玛", "去死", "草泥马", "滚"]}  # 需要传递的列表信息
data = {"contents": ["你这种行为是死妈行为"]}  # 需要传递的列表信息
data = json.dumps(data)
response = requests.post(url, data=data)
result2 = response.text
result2 = eval(result2)["embedding_result"]
from sklearn.metrics.pairwise import cosine_similarity

print(np.mean(cosine_similarity(np.array(result), np.array(result2)), axis=1))
