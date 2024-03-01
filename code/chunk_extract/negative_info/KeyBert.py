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
    for i in range(len(words)):
        for j in range(i + 1,i + 1+ n):
            ngram = ''.join(words[i:i + j])  # 创建一个n-gram
            if ngram not in ngrams:
                ngrams.append(ngram)

    return ngrams


# print(result)
def chunk_extract(text, embedding_type="pool"):
    # data = {"contents": text}  # 需要传递的列表信息
    # data = json.dumps(data)
    # response = requests.post(url, data=data)
    # result = response.text
    # result = eval(result)["embedding_result"] #doc embedding
    words_embedding_bags = []
    orig = text.replace(" ", "").replace("。", " ").replace("，", " ").replace("？", "").strip()
    orig = orig.split(" ")
    pre_url = "http://127.0.0.1:4567/soft_match/text2embedding?"
    # doc embedding
    content_bag = []
    doc_url = pre_url + f"contents={text}&embedding_type={embedding_type}"
    r = requests.get(url=doc_url)
    result_json = json.loads(r.text)
    doc_embed = result_json["embedding_result"]

    final_result = []

    for txt in orig:
        content_bag += create_ngram_model(txt, 3)
    # print(content_bag)
    # exit()
    for txt in content_bag:
        words_url = pre_url + f"contents={txt}&embedding_type={embedding_type}"
        r = requests.get(url=words_url)
        result_json = json.loads(r.text)
        words_embed = result_json["embedding_result"]
        words_embedding_bags.append(words_embed[0])
    print(np.array(words_embedding_bags).shape)
    print(np.array(doc_embed).shape)
    score = np.mean(cosine_similarity(np.array(words_embedding_bags), np.array(doc_embed)), axis=1)
    for text, value in zip(content_bag, score):
        final_result.append((text, value))
    sorted_list = sorted(final_result, key=lambda t: t[1], reverse=True)
    return sorted_list


if __name__ == '__main__':
    text = "打倒中共共产党，打倒中共,法轮功万岁，。"
    print(chunk_extract(text, embedding_type="pool"))
