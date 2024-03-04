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
import os

path = os.getcwd()
prepath = os.path.dirname(path)
print(path)
dim = 768
num_elements = 100000
from tqdm import tqdm


def index_update(vector_path):
    embeddings = pickle.load(open(vector_path, "rb"))
    embeddings = [k["vector"] for k in list(embeddings.values())]
    print("embeddings shape",len(embeddings[0]))
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)
    p.set_ef(10)
    p.set_num_threads(4)
    print("Adding first batch of %d elements" % (len(embeddings)))
    p.add_items(embeddings)
    print(prepath)
    p.save_index(prepath + r"\ir\softmatch\vector_index.bin")
    return {"update result": "update index successful"}


def search(index_model, embedding, sentences, labels, k):
    final_result = []
    targets, distances = index_model.knn_query(embedding, k)  # K=4
    print(targets, distances)
    for target, distance in zip(targets[0], distances[0]):
        final_result.append([sentences[target], labels[target], float(distance)])
    return {"search_result": final_result}


if __name__ == '__main__':
    #     num_elements = 100000
    from tqdm import tqdm
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    embeddings = pickle.load(open(r"D:\work\QiJi\qiji_compet\code\ir\vector.pkl", "rb"))
    vector = [k["vector"] for k in list(embeddings.values())]
    labels=[k["label"] for k in  list(embeddings.values())]
    contents = [k for k in list(embeddings.keys())]
    # print(vector[0])
    #
    # print(
    #     cosine_similarity(np.array(vector[0]).reshape(1, -1),
    #                       np.array(vector[522]).reshape(1, -1)))
    # print(labels[0],contents[0])
    # print(labels[522], contents[522])

#     p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
#     p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)
#     p.set_ef(10)
#     p.set_num_threads(4)
#     print("Adding first batch of %d elements" % (len(embeddings)))
#     p.add_items(embeddings, np.arange(len(embeddings)))
#     # hnsw model
    index_path = r"D:\work\QIJI\qiji_compet\code\ir\softmatch\vector_index.bin"
    p = hnswlib.Index(space='cosine', dim=768)
    p.load_index(index_path)
#     # Declaring index
#     print(len(embeddings[0]))
    targets, distances = p.knn_query(vector[0], 10)  # K=4
#     file_path = r"D:\work\QIJI\qiji_compet\code\data\dataset\multi_cls_data\train_multi_v2.xlsx"
#     data_init = pd.read_excel(file_path)
#     sentences = data_init["content"].values.tolist()
#     labels = data_init["label"].values.tolist()
#     #     sentence_bag, label_bag, distance_bag = [], [], []
    print(labels[0], contents[0])
    for i, (target, distance) in enumerate(zip(targets[0], distances[0])):
        print(target, distance, contents[target], labels[target])
# # vector_path=r"D:\work\QiJi\qiji_compet\code\ir\softmatch\vector.pkl"
# # index_update(vector_path)
