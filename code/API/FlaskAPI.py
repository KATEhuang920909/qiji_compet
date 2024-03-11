from flask import Flask, request
import sys
import os

sys.path.append("../")
sys.path.append("../ir/hardmatch")
sys.path.append("../ir/SentenceTransformer")
sys.path.append("../ir/softmatch")
sys.path.append("../chunk_extract/personal_info")
sys.path.append("../ir/dicts")

from Embedding import embedding
from dfa import DFA
from search import SEARCH
import paddle
from MatchModel import SentenceTransformer
import pandas as pd
import hnswlib
from paddlenlp.transformers import ErnieTokenizer, ErnieModel
import numpy as np
from tqdm import tqdm
from paddlenlp.data import Pad, Stack, Tuple
from predict_bigru_crf import load_dict, convert_tokens_to_ids, Predictor, args
import pickle
import jieba

app = Flask(__name__)
current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.dirname(current_path)
# hard match model
dfa = DFA()

##### softmatch model
bm25model_path = parent_path + r"\models\search_model\bm25model.pickle"
bm25model = pickle.load(open(bm25model_path, 'rb'))
params_path = parent_path + r"\models\embedding_model"
tokenizer = ErnieTokenizer.from_pretrained(params_path)
embedding_model = ErnieModel.from_pretrained(params_path)
embedding_model.eval()


def load_hnsw_model(parent_path):
    # hnsw model
    index_path0 = parent_path + r"\models\search_model\vector_index0.bin"
    index0 = hnswlib.Index(space='cosine', dim=768)
    index0.load_index(index_path0)

    index_path1 = parent_path + r"\models\search_model\vector_index1.bin"
    index1 = hnswlib.Index(space='cosine', dim=768)
    index1.load_index(index_path1)

    index_path2 = parent_path + r"\models\search_model\vector_index2.bin"
    index2 = hnswlib.Index(space='cosine', dim=768)
    index2.load_index(index_path2)

    index_path3 = parent_path + r"\models\search_model\vector_index3.bin"
    index3 = hnswlib.Index(space='cosine', dim=768)
    index3.load_index(index_path3)

    index_path4 = parent_path + r"\models\search_model\vector_index4.bin"
    index4 = hnswlib.Index(space='cosine', dim=768)
    index4.load_index(index_path4)
    return index0, index1, index2, index3, index4


def load_cont2lb_model(parent_path):
    # hnsw model
    cont2lb_path0 = parent_path + r"\models\search_model\content2label0.pickle"
    cont2lb0 = pickle.load(open(cont2lb_path0, 'rb'))

    cont2lb_path1 = parent_path + r"\models\search_model\content2label0.pickle"
    cont2lb1 = pickle.load(open(cont2lb_path1, 'rb'))

    cont2lb_path2 = parent_path + r"\models\search_model\content2label0.pickle"
    cont2lb2 = pickle.load(open(cont2lb_path2, 'rb'))

    cont2lb_path3 = parent_path + r"\models\search_model\content2label0.pickle"
    cont2lb3 = pickle.load(open(cont2lb_path3, 'rb'))

    cont2lb_path4 = parent_path + r"\models\search_model\content2label0.pickle"
    cont2lb4 = pickle.load(open(cont2lb_path4, 'rb'))
    return cont2lb0, cont2lb1, cont2lb2, cont2lb3, cont2lb4


# index0, index1, index2, index3, index4 = load_hnsw_model(parent_path)
# cont2lb0, cont2lb1, cont2lb2, cont2lb3, cont2lb4 = load_cont2lb_model(parent_path)
# print("Loaded parameters from %s" % params_path)


## chunk extract model


# ===============hard match====================
@app.route('/hard_match/filter', methods=['POST', 'GET'])
def text_filter():
    string = request.args.get('contents', '')
    response_dict = dict()
    print("string", string)
    if dfa.exists(string) is False:
        response_dict['is_illegal'] = False
        position = []
    else:
        response_dict['is_illegal'] = True
        position = dfa.filter_all(string)
    response_dict['position'] = position
    print("response_dict", response_dict)

    # response = json.dumps(response_dict)
    return response_dict


@app.route('/hard_match/add', methods=['POST', 'GET'])
def add_new_words():
    string = request.args.get('word', '')
    label = request.args.get('label', '')
    if string in dfa.ban_words_set:
        return '"' + string + '"已在敏感词文档中，添加失败'
    dfa.add_new_word(string, label)
    with open(dfa.path, 'a', encoding='utf-8-sig') as f:
        f.writelines('\n' + string)
    return '添加成功'


@app.route('/hard_match/change', methods=['POST', 'GET'])
def chang_text():
    path = request.args.get('path', '')
    try:
        dfa.change_words(path)
    except FileNotFoundError:
        return '文件"' + path + '"不存在'
    return '已将文件"' + path + '"作为敏感词库'


# ===============soft match====================
@app.route('/soft_match/text2embedding', methods=['POST', 'GET'])
def text2embedding():
    # if request.method == 'POST':
    #     text = request.data
    #     text = list(eval(text.decode("unicode_escape")).values())[0]  # lists
    # else:
    text = request.args.get('contents')
    try:
        text = eval(text)
    except:
        pass
    results = embedding(embedding_model, text, tokenizer)
    # print(results)
    return results


@app.route('/soft_match/vector_update', methods=['POST', 'GET'])
def vector_update():
    file_path = request.args.get('file_path', '')
    save_path = request.args.get('save_path', '')
    data = pd.read_excel(file_path)  # .sample(n=200)
    sentences = data["content"].values.tolist()
    labels = data["label"].values.tolist()
    content_bag = []
    for i, (con, lb) in tqdm(enumerate(zip(sentences, labels))):
        result = embedding(embedding_model, con, tokenizer)
        content_bag.append({"label": lb, "vector": result[0]})
    content2embed = dict(zip(sentences, content_bag))  # {content1:{"label":,"vector":},content2:{}...}
    try:
        pickle.dump(content2embed, open(save_path + r"\vector.pkl", "wb"))
        return {"embedding_result": "update vector successful"}
    except Exception as e:
        return {"embedding_result": str(e)}


@app.route('/soft_match/index_update', methods=['POST', 'GET'])
def Index_Update():
    vector_path = request.args.get('vector_path', '')
    embeddings_dic = pickle.load(open(vector_path, "rb"))
    contents = list(embeddings_dic.keys())
    labels = [k["label"] for k in list(embeddings_dic.values())]
    embeddings = [k["vector"] for k in list(embeddings_dic.values())]

    # 分桶更新
    retult = SEARCH.index_update(contents, labels, embeddings)

    contents_cut = [jieba.lcut(x) for x in contents]
    retult2 = SEARCH.bm25_update(contents_cut)
    return retult, retult2


@app.route('/soft_match/search', methods=['POST', 'GET'])
def Search():
    text = request.args.get('text', '')
    k = int(request.args.get('topk', ''))
    print(text)
    vector = embedding(embedding_model, text, tokenizer)
    text_cut = jieba.lcut(text)
    if 0 < len(text) <= 10:
        vec_result = SEARCH.search_vec(index0, vector, list(cont2lb0.keys()), list(cont2lb0.values()), k)

        bm25_result = SEARCH.search_bm25(bm25model, text_cut, list(cont2lb0.keys()), list(cont2lb0.values()), 5)
    elif 10 < len(text) <= 20:
        vec_result = SEARCH.search_vec(index0, vector, list(cont2lb1.keys()), list(cont2lb1.values()), k)
        bm25_result = SEARCH.search_bm25(bm25model, text_cut, list(cont2lb1.keys()), list(cont2lb1.values()), 5)
    elif 20 < len(text) <= 40:
        vec_result = SEARCH.search_vec(index0, vector, list(cont2lb2.keys()), list(cont2lb2.values()), k)
        bm25_result = SEARCH.search_bm25(bm25model, text_cut, list(cont2lb2.keys()), list(cont2lb2.values()), 5)
    elif 40 < len(text) <= 100:
        vec_result = SEARCH.search_vec(index0, vector, list(cont2lb3.keys()), list(cont2lb3.values()), k)
        bm25_result = SEARCH.search_bm25(bm25model, text_cut, list(cont2lb3.keys()), list(cont2lb3.values()), 5)
    else:
        vec_result = SEARCH.search_vec(index0, vector, list(cont2lb4.keys()), list(cont2lb4.values()), k)
        bm25_result = SEARCH.search_bm25(bm25model, text_cut, list(cont2lb4.keys()), list(cont2lb4.values()), 5)
    vec_result.update(bm25_result)
    return vec_result

# ===============personal info extract====================
# @app.route('/ner/person_info_check', methods=['POST', 'GET'])
# def ner_predict():
#     # if request.method == 'POST':
#     #     text = request.data
#     #     text = list(eval(text.decode("unicode_escape")).values())[0]  # lists
#     # else:
#     text = request.args.get('contents')
#     if text.strip():
#         text = text.strip()
#         len_text = [[len(text)]]
#
#         results = predictor.predict([[text, len_text]], batchify_fn, word_vocab, label_vocab)
#     print(results)
#     return {"ner_result": results}


if __name__ == '__main__':
    port = 4567
    app.run('0.0.0.0', port)
