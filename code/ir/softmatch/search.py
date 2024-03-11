# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 19:57
# @Author  : huangkai
# @File    : faiss_experiment.py
import hnswlib
import numpy as np
# import pandas as pd
from collections import Counter
import time
import os
from gensim.summarization import bm25
import pickle

dim = 768
num_elements = 100000
from tqdm import tqdm


class SEARCH():
    def __init__(self):
        path = os.getcwd()
        prepath = os.path.dirname(path)
        self.save_path = prepath + r"\models\search_model"

    def len_sep(self, text1, k1, k2):
        if len(text1) > k1 and len(text1) <= k2:
            return True
        return False

    def index_update(self, contents, labels, embeddings):

        content010, content1020, content2040, content40100, content100 = {}, {}, {}, {}, {}
        embedding010, embedding1020, embedding2040, embedding40100, embedding100 = [], [], [], [], []
        for i, cont, embed in enumerate(zip(contents, embeddings)):
            if self.len_sep(cont, 0, 10):
                content010[cont] = labels[i]
                embedding010.append(embed)
            if self.len_sep(cont, 10, 20):
                content1020[cont] = labels[i]
                embedding1020.append(embed)
            if self.len_sep(cont, 20, 40):
                content2040[cont] = labels[i]
                embedding2040.append(embed)
            if self.len_sep(cont, 40, 100):
                content40100[cont] = labels[i]
                embedding40100.append(embed)
            if self.len_sep(cont, 100, 1e8):
                content100[cont] = labels[i]
                embedding100.append(embed)
        for i, embed, cont in enumerate(zip([embedding010, embedding1020, embedding2040, embedding40100, embedding100]
                                            [content010, content1020, content2040, content40100, content100])):
            p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
            num_elements = len(embed)
            p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)
            p.set_ef(10)
            p.set_num_threads(4)
            print("Adding first batch of %d elements" % (len(embed)))
            p.add_items(embed)
            p.save_index(self.save_path + rf"\vector_index{i}.bin")
            with open(self.save_path + rf'\content2label{i}.pickle', 'wb') as f:
                pickle.dump(cont, f)
        return {"update result": "update index successful"},

    def bm25_update(self, content_cut):
        bm25Model = bm25.BM25(content_cut)  # 字符级bm25
        # 将Python对象保存到pickle文件中
        with open(self.save_path + r'\bm25model.pickle', 'wb') as f:
            pickle.dump(bm25Model, f)
        return {"update result": "update bm25 successful"}

    def search_vec(self, index_model, embedding, sentences, labels, k=10):
        final_result = []
        targets, distances = index_model.knn_query(embedding, k)  # K=4
        for target, distance in zip(targets[0], distances[0]):
            final_result.append([sentences[target], labels[target], float(distance)])
        return {"vec_search_result": final_result}

    def search_bm25(self, bm25_model, cut_sentences: list, sentences, labels, k=5):
        final_result = []
        raw_rank_index = np.array(bm25_model.get_scores(cut_sentences)).argsort().tolist()[::-1][1:k + 1]  # bm25的结果
        temp_content = [sentences[k] for k in raw_rank_index]
        temp_labels = [labels[k] for k in raw_rank_index]
        for content, label in zip(temp_content, temp_labels):
            final_result.append([content, label])

        return {"bm25_search_result": final_result}


if __name__ == '__main__':
    #     num_elements = 100000
    from tqdm import tqdm
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    # embeddings = pickle.load(open(r"D:\work\QiJi\qiji_compet\code\ir\vector.pkl", "rb"))
    # vector = [k["vector"] for k in list(embeddings.values())]
    # labels = [k["label"] for k in list(embeddings.values())]
    # contents = [k for k in list(embeddings.keys())]
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
