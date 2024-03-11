import argparse
import json
import os
import random

import pandas as pd
from gensim.summarization import bm25
import jieba
import numpy as np
from tqdm import tqdm
from collections import Counter
path = "D:\work\qiji_compet\code\data\dataset\multi_cls_data"
train = pd.read_excel(path + r"\train_multi.xlsx")
dev = pd.read_excel(path + r"\dev_multi.xlsx")
data = pd.concat([train[["content", "labels"]], dev[["content", "labels"]]])
content = data.content.values
label=data.labels.values

content_cut = [jieba.lcut(k) for k in content]  # 案件基本情况

bm25Model = bm25.BM25(content_cut)  # 字符级bm25

import pickle as pkl

import pickle


# 将Python对象保存到pickle文件中
with open('bm25.pickle', 'wb') as f:
    pickle.dump(bm25Model, f)

# 从pickle文件中读取Python对象
with open('bm25.pickle', 'rb') as f:
    bm25Model = pickle.load(f)


# # bm25准确性
test = pd.read_excel(path + r"\test_multi.xlsx")
content_test = test.content.values # 案件基本情况
content_test_cut = [jieba.lcut(k)  for k in content_test]  # 案件基本情况
final_content,final_label =[],[]
for q in tqdm(content_test):
    raw_rank_index = np.array(bm25Model.get_scores(q)).argsort().tolist()[::-1][1:10]  # bm25的结果
    temp_content = [content[k] for k in raw_rank_index]
    labels=[label[k] for k in raw_rank_index]
    label_counts = Counter(labels).items()
    sorted_counts = sorted(label_counts, key=lambda x: x[1], reverse=True)
    temp_label = sorted_counts[0][0]
    final_content.append(temp_content)
    final_label.append(temp_label)
test["final_content"]=final_content
test["final_label"]=final_label
test.to_excel("test_bm25.xlsx")
