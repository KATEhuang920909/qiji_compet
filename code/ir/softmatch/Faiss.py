# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 19:57
# @Author  : huangkai
# @File    : faiss_experiment.py
import hnswlib
import numpy as np
# import pandas as pd
import pickle
from collections import Counter
import time

dim = 768
num_elements = 100000
from tqdm import tqdm


# data = pd.read_excel(r"D:\work\骐骥杯\qiji_compet\code\data\dataset\multi_cls_data\train_multi_v2.xlsx")
# filters = "[^a-zA-Z\u4e00-\u9fd5]"
# sentences = data["content"].tolist()
# labels = data["label"].tolist()


def index_update(vector_path):
    embeddings = pickle.load(open(vector_path, "rb"))
    print(list(embeddings.values())[0])
    embeddings = [k["vector"] for k in list(embeddings.values())]
    print(embeddings[0],type(embeddings[0]))
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)
    p.set_ef(10)
    p.set_num_threads(4)
    print("Adding first batch of %d elements" % (len(embeddings)))
    p.add_items(embeddings, np.arange(len(embeddings)))
    p.save_index(r"D:\code\qiji_compet\code\models\vector_index")
    return {"update result": "update index successful"}

def search(index_model, embedding, sentences, labels, k):
    final_result = []
    targets, distances = index_model.knn_query(embedding, k)  # K=4
    for target, distance in zip(targets[0], distances[0]):
        final_result.append([sentences[target], labels[target], float(distance)])
    return {"search_result": final_result}
if __name__ == '__main__':
    num_elements = 100000
    from tqdm import tqdm
    embeddings = pickle.load(open(r"D:\code\qiji_compet\code\models\vector.pkl", "rb"))
    embeddings = [k["vector"] for k in list(embeddings.values())]
    # p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
    # p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)
    # p.set_ef(10)
    # p.set_num_threads(4)
    # print("Adding first batch of %d elements" % (len(embeddings)))
    # p.add_items(embeddings, np.arange(len(embeddings)))
    # hnsw model
    index_path = r"D:\code\qiji_compet\code\models\vector_index"
    p = hnswlib.Index(space='cosine', dim=768)
    p.load_index(index_path)
    # Declaring index
    print(len(embeddings[0]))
    targets, distances = p.knn_query(embeddings[0], 5)  # K=4
    sentence_bag, label_bag, distance_bag = [], [], []
    for target, distance in zip(targets[0], distances[0]):
        print(target,distance)