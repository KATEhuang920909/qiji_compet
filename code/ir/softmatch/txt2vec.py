# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     text2vec
   Author :       huangkai
   date：          2024/2/19
-------------------------------------------------
"""
from text2vec import SentenceModel
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']

model = SentenceModel(r'D:\code\qiji_compet\code\models\text2vec')
embeddings = model.encode(sentences)
print(embeddings)