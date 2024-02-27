# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     KeyBert
   Author :       huangkai
   date：          2024/2/25
-------------------------------------------------
"""
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import numpy as np
import json
import requests


def create_ngram_model(text, n):
    words = jieba.lcut(text)
    ngrams = []  # 用于存储n-grams的列表

    for i in range(len(words) - n + 1):
        for j in range(1, n + 1):
            ngram = ''.join(words[i:i + j])  # 创建一个n-gram
            ngrams.append(ngram)

    return ngrams


# print(result)
def chunk_extract(text):
    # data = {"contents": text}  # 需要传递的列表信息
    # data = json.dumps(data)
    # response = requests.post(url, data=data)
    # result = response.text
    # result = eval(result)["embedding_result"] #doc embedding
    content_bags = []
    orig = text.replace(" ", "").replace("。", " ").replace("，", " ").replace("？", "").strip()
    orig = orig.split(" ")

    # doc embedding
    txts = {"contents": [text]}  # 需要传递的doc信息
    content_bag = []
    url = r"http://127.0.0.1:4567/soft_match/text2embedding"
    txts = json.dumps(txts)
    doc_embed = requests.post(url, data=txts)
    doc_embed = doc_embed.text
    doc_embed = eval(doc_embed)["embedding_result"]
    final_result = []
    for txt in orig:
        content_bag += create_ngram_model(txt, 3)

    data = {"contents": content_bag}  # 需要传递的列表信息
    # word bag embedding
    data = json.dumps(data)
    response = requests.post(url, data=data)
    word_embed = response.text
    word_embed = eval(word_embed)["embedding_result"]

    score = np.mean(cosine_similarity(np.array(word_embed), np.array(doc_embed)), axis=1)
    for text, value in zip(content_bag, score):
        final_result.append((text, value))
    sorted_list = sorted(final_result, key=lambda t: t[1], reverse=True)
    return sorted_list


if __name__ == '__main__':
    text = "选择远离中共避免成为天灭中共时的陪葬更多真相URL"
    print(chunk_extract(text))
