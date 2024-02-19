# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 19:57
# @Author  : huangkai
# @File    : faiss_experiment.py
import hnswlib
import numpy as np
import pandas as pd
import pickle

dim = 768
num_elements = 100000

from text2vec import SentenceModel

data = pd.read_excel(r"D:\code\qiji_compet\code\data\dataset\multi_cls_data\train_multi.xlsx")
filters = "[^a-zA-Z\u4e00-\u9fd5]"
sentences = data["content"].unique().tolist()[:100]

model = SentenceModel(r'D:\code\qiji_compet\code\models\text2vec')
embeddings = model.encode(sentences)
# print(embeddings)
content2embed = dict(zip(sentences, embeddings))
pickle.dump(content2embed, open("vector.pkl", "wb"))

p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)
p.set_ef(10)
p.set_num_threads(4)
print("Adding first batch of %d elements" % (len(embeddings)))
p.add_items(embeddings, np.arange(len(embeddings)))

# Declaring index

embedding = model.encode("彩票集团URL为您打造最大网络彩票投注平台视频同步赌场游戏让您随时随地畅享博彩乐趣")
# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(embedding, 4)  # K=4
for target, distance in zip(labels, distances):
    for t,d in zip(target[1:],distance[1:]):
        line = "{}:{}".format(sentences[target[0]],sentences[t])
        print(line,d)
#
# # Serializing and deleting the index:
# index_path='first_half.bin'
# print("Saving index to '%s'" % index_path)
# p.save_index("first_half.bin")
# del p
#
# # Re-initializing, loading the index
# p = hnswlib.Index(space='cosine', dim=dim)  # the space can be changed - keeps the data, alters the distance function.
#
# print("\nLoading index from 'first_half.bin'\n")
#
# # Increase the total capacity (max_elements), so that it will handle the new data
# p.load_index("first_half.bin", max_elements = num_elements)


# if __name__ == "__main__":
#     # time_test()
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import pickle
#     from text2vec import SentenceModel
#
#     data = pd.read_excel(r"D:\code\qiji_compet\code\data\dataset\multi_cls_data\train_multi.xlsx")
#     filters = "[^a-zA-Z\u4e00-\u9fd5]"
#     sentences = data["content"].unique().tolist()[:5]
#
#     model = SentenceModel(r'D:\code\qiji_compet\code\models\text2vec')
#     embeddings = model.encode(sentences)
#     # print(embeddings)
#     content2embed = dict(zip(sentences, embeddings))
#     pickle.dump(content2embed,open("vector.pkl","wb"))
