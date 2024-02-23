# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 19:57
# @Author  : huangkai
# @File    : faiss_experiment.py
import hnswlib
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import time

dim = 768
num_elements = 100000
from tqdm import tqdm
from text2vec import SentenceModel

data = pd.read_excel(r"D:\work\骐骥杯\qiji_compet\code\data\dataset\multi_cls_data\train_multi_v2.xlsx")
filters = "[^a-zA-Z\u4e00-\u9fd5]"
sentences = data["content"].tolist()
labels = data["label"].tolist()
model = SentenceModel(r'D:\work\骐骥杯\qiji_compet\code\models\text2vec')
# embeddings = model.encode(sentences)
# print(embeddings)
# content2embed = dict(zip(sentences, embeddings))
# pickle.dump(content2embed, open("vector.pkl", "wb"))
embeddings = pickle.load(open("vector.pkl", "rb"))
embeddings=list(embeddings.values())
p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)
p.set_ef(10)
p.set_num_threads(4)
print("Adding first batch of %d elements" % (len(embeddings)))
p.add_items(embeddings, np.arange(len(embeddings)))

# Declaring index
# test_data = pd.read_excel(r"D:\work\骐骥杯\qiji_compet\code\data\dataset\multi_cls_data\dev_multi_v2.xlsx")
# test_sentences = test_data["content"].tolist()
# test_labels = test_data["label"].tolist()
test_data = open(r"D:\work\骐骥杯\qiji_compet\code\data\dev.txt",encoding="utf8").readlines()
test_data = [k.strip().split("\t") for k in test_data]
test_sentences = [k[1] for k in test_data]
test_labels = [int(k[0][-1]) for k in test_data]
final_result = []
t1 = time.time()
for i, text in tqdm(enumerate(test_sentences)):
    embedding = model.encode(text)
    # Query the elements for themselves and measure recall:
    targets, distances = p.knn_query(embedding, 5)  # K=4
    sentence_bag, label_bag, distance_bag = [], [], []
    for target, distance in zip(targets[0], distances[0]):
        sentence_bag.append(sentences[target])
        label_bag.append(labels[target])
        distance_bag.append(distance)
    final_label = Counter(label_bag)
    final_label = max(final_label, key=final_label.get)
    line = (text, sentence_bag, label_bag, final_label, test_labels[i], distance_bag)
    final_result.append(line)
print(time.time() - t1, (time.time() - t1) / len(final_result))
final_data = pd.DataFrame(final_result,
                          columns=["text", "sentence_bag", "label_bag", "final_label", "true_label", "score"])
final_data["result"] = final_data.final_label == final_data.true_label
final_data.to_excel("final_data_v4.xlsx")
