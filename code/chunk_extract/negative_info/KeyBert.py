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
    word_index = []
    # print(words)
    for i in range(len(words)):
        for j in range(1, n + 1):
            ngram = ''.join(words[i:i + j])  # 创建一个n-gram
            if ngram not in ngrams:
                ngrams.append(ngram)
                start_index = len(''.join(words[:i]))
                word_index.append((start_index, start_index + len(ngram)))

    return ngrams, word_index


# print(result)
def chunk_extract(text, orig):
    # data = {"contents": text}  # 需要传递的列表信息
    # data = json.dumps(data)
    # response = requests.post(url, data=data)
    # result = response.text
    # result = eval(result)["embedding_result"] #doc embedding
    words_embedding_bags = []

    pre_url = "http://127.0.0.1:4567/soft_match/text2embedding?"
    # doc embedding
    content_bag = []
    doc_url = pre_url + f"contents={text}"
    r = requests.get(url=doc_url)
    result_json = json.loads(r.text)
    doc_embed = result_json["embedding_result"]

    final_result = []
    pre_index = 0
    for k, txt in enumerate(orig):
        if k != 0:
            pre_index += len(txt)
        content, index = create_ngram_model(txt, 3)

        index = [[k + pre_index, v + pre_index] for (k, v) in index]
        # print(txt, content, index)
        # print("\n")
        for con, idx in zip(content, index):
            content_bag.append({"text": "".join(orig), "chunk": con, "index": idx})
    chunk_txt = [k["chunk"] for k in content_bag]
    for txt in chunk_txt:
        words_url = pre_url + f"contents={txt}"
        r = requests.get(url=words_url)
        result_json = json.loads(r.text)
        words_embed = result_json["embedding_result"]
        words_embedding_bags.append(words_embed[0])
    score = np.mean(cosine_similarity(np.array(words_embedding_bags), np.array(doc_embed)), axis=1)
    for i, (text_dic, value) in enumerate(zip(content_bag, score)):
        final_result.append((text_dic["text"], text_dic["chunk"], text_dic["index"], value))
    sorted_list = sorted(final_result, key=lambda t: t[-1], reverse=True)
    if sorted_list[0][-1] < 0.9:
        return sorted_list[0]
    return [(i, j) for i, j in sorted_list if j >= 0.9]


if __name__ == '__main__':
    text = ".,,.[;[.],[.],.[],.["
    orig = text.replace(" ", "").replace("。", " ").replace("，", " ").replace("？", "").strip()
    orig = orig.split(" ")
    # print(orig)
    print(chunk_extract(text, orig))
