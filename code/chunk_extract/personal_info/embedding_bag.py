"""
embedding提取器
词表为ernie3.0的词表，

"""

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     embedding
   Author :       huangkai
   date：          2024/2/24
-------------------------------------------------
"""
import argparse
import numpy
import numpy as np
import paddle
from paddlenlp.data import Pad, Tuple
from tqdm import tqdm
from paddlenlp.transformers import AutoModel, AutoTokenizer
import sys
sys.path.append("../../ir/sentence_transformers")
from Embedding import convert_example,embedding
from model import SentenceTransformer
from tqdm import tqdm
def vectorizer(input, label=None, length=2000):
    if label is not None:
        for x, y in zip(input, label):
            yield np.array((x + [0]*length)[:length]).astype('int64'), np.array([y]).astype('int64')
    else:
        for x in input:
            yield np.array((x + [0]*length)[:length]).astype('int64')


# ErnieTinyTokenizer is special for ernie-tiny pretained model.
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
data = [
    "世界上什么东西最小",
    "光眼睛大就好看吗",
    "小蝌蚪找妈妈怎么样",
]
vocab=open(r"D:\work\QiJi\qiji_compet\code\chunk_extract\personal_info\data\word.dic",
           "r",encoding="utf8").readlines()
vocab=[k.strip() for k in vocab]
print(vocab[:10])
pretrained_model = AutoModel.from_pretrained(r"ernie-3.0-medium-zh")
model = SentenceTransformer(pretrained_model)


state_dict = paddle.load("D:\work\QiJi\qiji_compet\code\ir\softmatch\match_model\model_state.pdparams")
model.set_dict(state_dict)
print("Loaded parameters from %s" %"D:\work\QiJi\qiji_compet\code\ir\softmatch\match_model\model_state.pdparams")

results = embedding(model, vocab, tokenizer)
for idx, text in tqdm(enumerate(vocab)):
    print("Data: {} \t Embedding: {}".format(text, results["embedding_result"][idx]))
#+++++++++++++++++++++++save+++++++++++++++++++++++
import json
import pickle

# JSON数据
data = dict(zip(vocab,results["embedding_result"]))

# 转换为字符串形式
json_str = json.dumps(data)

# 将JSON字符串写入到二进制文件
with open("vocab2vector.bin", "wb") as file:
    pickle.dump(json_str, file)
# import pickle
# import json
# from sklearn.metrics.pairwise import cosine_similarity
# data2=pickle.load(open("vocab2vector.bin","rb"))
# data2=json.loads(data2)
# for unit in data2:
#     print(cosine_similarity(np.array(data[unit]).reshape(1, -1),
#                             np.array(data2[unit]).reshape(1, -1)))