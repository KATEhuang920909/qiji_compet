# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 19:57
# @Author  : huangkai
# @File    : faiss_experiment.py
import hnswlib
import jieba
import numpy as np
# import pandas as pd
from collections import Counter
import time
import os
from lmir import LMIR
import pandas as pd
from gensim.summarization import bm25
import pickle

dim = 768
max_elements = 1000000
from tqdm import tqdm


class SEARCH():
    def __init__(self):

        path = os.getcwd()
        prepath = os.path.dirname(path)
        self.save_path = prepath + r"\models\search_model"
        self.knowledge_base = prepath + r"\data\knowledge_data\knowledge_base.xlsx"
        print("Load index from %s" % self.save_path)
        self.hnsw_model_bag = self.load_hnsw_model(self.save_path)
        # index0, index1, index2, index3, index4 = hnsw_model_bag

        print("load content label from %s" % self.save_path)
        self.cont2lb_bag = self.load_cont2lb_model(self.save_path)
        # cont2lb0, cont2lb1, cont2lb2, cont2lb3, cont2lb4 = cont2lb_bag
        self.knowledge_base = prepath + r"\data\knowledge_data\knowledge_base.xlsx"
        knowledge = pd.read_excel(self.knowledge_base)

        self.knowledge_content = knowledge.content.values
        self.knowledge_label = knowledge.label.values

        # update lmir
        contents_cut = [list(x) for x in self.knowledge_content]
        self.bm25_model = bm25.BM25(contents_cut)
        with open(self.save_path + r'\bm25model.pickle', 'wb') as f:
            pickle.dump(self.bm25_model, f)
        print({"update bm25 result": "update bm25 successful"})

    def load_hnsw_model(self, parent_path):
        # hnsw model
        hnsw_model_bag = []
        for i in range(5):
            index_path = parent_path + rf"/index{i}.bin"
            index = hnswlib.Index(space='cosine', dim=768)
            index.load_index(index_path)
            hnsw_model_bag.append(index)
            print(f"load index{i} done")

        return hnsw_model_bag

    def load_cont2lb_model(self, parent_path):
        # hnsw model
        cont2lb_bag = []
        for i in range(5):
            cont2lb_path = parent_path + rf"/content2label{i}.pickle"
            cont2lb = pickle.load(open(cont2lb_path, 'rb'))
            cont2lb_bag.append(cont2lb)
            print(f"load cont2lb{i} done")
        return cont2lb_bag

    def len_sep(self, text1, k1, k2):
        if len(text1) > k1 and len(text1) <= k2:
            return True
        return False

    def index_update(self, contents, labels, embeddings):

        content010, content1020, content2040, content40100, content100 = {}, {}, {}, {}, {}
        embedding010, embedding1020, embedding2040, embedding40100, embedding100 = [], [], [], [], []
        for i, (cont, embed) in enumerate(zip(contents, embeddings)):
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
        for i, (embed, cont) in enumerate(
                zip([embedding010, embedding1020, embedding2040, embedding40100, embedding100],
                    [content010, content1020, content2040, content40100, content100])):
            num_elements = len(embed)
            if num_elements != 0:
                p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip

                p.init_index(max_elements=max_elements // 2, ef_construction=100, M=16)
                p.set_ef(10)
                p.set_num_threads(4)
                print("Adding first batch of %d elements" % (len(embed)))
                p.add_items(embed, np.arange(len(embed)))
                p.save_index(self.save_path + rf"\vector_index{i}.bin")
                pickle.dump(cont, open(self.save_path + rf'\content2label{i}.pickle', 'wb'))
        return {"update index result": "update index successful"}

    # def bm25_update(self, content_cut):
    #     with open(self.save_path + r'\bm25model.pickle', 'wb') as f:
    #         pickle.dump(bm25Model, f)
    #     return {"update bm25 result": "update bm25 successful"}

    def search_vec(self, index_model, embedding, sentences, labels, k=10):
        final_result = []
        targets, distances = index_model.knn_query(embedding, k)  # K=4
        for target, distance in zip(targets[0], distances[0]):
            final_result.append([sentences[target], labels[target], float(distance)])
        return {"vec_search_result": final_result}

    def search_bm25(self, cut_sentences: list, k=5):
        final_result = []
        raw_rank_index = np.array(self.bm25_model.get_scores(cut_sentences)).argsort().tolist()[::-1][
                         1:k + 1]  # bm25的结果
        temp_content = [self.knowledge_content[k] for k in raw_rank_index]
        temp_labels = [self.knowledge_label[k] for k in raw_rank_index]
        for content, label in zip(temp_content, temp_labels):
            final_result.append([content, label])

        return {"bm25_search_result": final_result}

    def search(self, vector, text, text_cut, k):
        bm25_result = self.search_bm25(text_cut,k)
        if 0 < len(text) <= 10:
            print("search index ,length betweem (0,10]")
            vec_result = self.search_vec(self.hnsw_model_bag[0], vector, list(self.cont2lb_bag[0].keys()),
                                         list(self.cont2lb_bag[0].values()), k)
        elif 10 < len(text) <= 20:
            print("search index ,length betweem (10,20]")
            vec_result = self.search_vec(self.hnsw_model_bag[1], vector, list(self.cont2lb_bag[1].keys()),
                                         list(self.cont2lb_bag[1].values()), k)

        elif 20 < len(text) <= 40:
            print("search index ,length betweem (20,40]")
            vec_result = self.search_vec(self.hnsw_model_bag[2], vector, list(self.cont2lb_bag[2].keys()),
                                         list(self.cont2lb_bag[2].values()), k)
        elif 40 < len(text) <= 100:
            print("search index ,length betweem (40,100]")
            vec_result = self.search_vec(self.hnsw_model_bag[3], vector, list(self.cont2lb_bag[3].keys()),
                                         list(self.cont2lb_bag[3].values()), k)
        else:
            print("search index ,length  (100,+)")
            vec_result = self.search_vec(self.hnsw_model_bag[4], vector, list(self.cont2lb_bag[4].keys()),
                                         list(self.cont2lb_bag[4].values()), k)
        print(bm25_result)
        print(vec_result)
        vec_result.update(bm25_result)
        return vec_result

# if __name__ == '__main__':
# embeddings = pickle.load(open(r"D:\work\qiji_compet\code\models\search_model\vector.pkl", "rb"))
# vector = [k["vector"] for k in list(embeddings.values())]
# label = [k["label"] for k in list(embeddings.values())]
# content = [k for k in list(embeddings.keys())]
# num=0
# index_path = rf"D:\work\qiji_compet\code\models\search_model\index{num}.bin"
# cont_lb_path= rf"D:\work\qiji_compet\code\models\search_model\content2label{num}.pickle"
# p = hnswlib.Index(space='cosine', dim=768)
# p.load_index(index_path)
#
# cont2lb = pickle.load(open(cont_lb_path, 'rb'))
# contents=list(cont2lb.keys())
# labels = list(cont2lb.values())
# #     # Declaring index
# #     print(len(embeddings[0]))
# sample=5
# targets, distances = p.knn_query(vector[sample], 10)  # K=4
# print(content[sample],label[sample])
# for i, (target, distance) in enumerate(zip(targets[0], distances[0])):
#     print(target, distance, contents[target], labels[target])

# parent_path = "D:\work\qiji_compet\code\models\search_model"
# hnsw_model_bag = load_hnsw_model(parent_path)
# index0, index1, index2, index3, index4 = hnsw_model_bag
# print("Loaded index  %s" % parent_path)
#
# cont2lb_bag = load_cont2lb_model(parent_path)
# cont2lb0, cont2lb1, cont2lb2, cont2lb3, cont2lb4 = cont2lb_bag
# print("content label from %s" % parent_path)
# index_path = rf"D:\work\qiji_compet\code\models\search_model\index0.bin"
# p = hnswlib.Index(space='cosine', dim=768)
# p.load_index(index_path)

# knowledge_base ="D:\work\qiji_compet\code\data\knowledge_data\knowledge_base.xlsx"
#
# knowledge = pd.read_excel(knowledge_base)
#
# knowledge_content = knowledge.content.values
# knowledge_label = knowledge.label.values

# cont2lb_path = rf"D:\work\qiji_compet\code\models\search_model\content2label0.pickle"
# cont2lb = pickle.load(open(cont2lb_path, 'rb'))
# contents=list(cont2lb.keys())
# labels = list(cont2lb.values())
# # print(len(knowledge_content),len(contents))
# # exit()
# print(contents[:2])
# contents_cut = [jieba.lcut(x) for x in contents]
# bm25Model = bm25.BM25(contents_cut)
# text = "枪支弹药"
# text_cut = jieba.lcut(text)
# # with open('bm25.pickle', 'rb') as f:
# #     bm25Model = pickle.load(f)
#
# raw_rank_index = np.array(bm25Model.get_scores(text_cut)).argsort().tolist()[::-1][:10]  # bm25的结果
# print(raw_rank_index[:10])
# print(bm25Model.get_scores(text_cut)[::-1])
# exit()
# temp_content = [contents[k] for k in raw_rank_index]
# temp_labels = [labels[k] for k in raw_rank_index]
# for cont, lb in zip(temp_content, temp_labels):
#     print(cont, lb)
# text = "枪支弹药"
# text_cut = list(text)
# search = SEARCH()
# print(search.search_bm25(text_cut))
