from flask import Flask, request
import sys

sys.path.append("../")
sys.path.append("../ir/hardmatch")
sys.path.append("../ir/sentence_transformers")
sys.path.append("../ir/softmatch")
from Embedding import embedding
from dfa import DFA
import json
from Faiss import index_update, search
import paddle
from model import SentenceTransformer
import pandas as pd
import hnswlib
from paddlenlp.transformers import AutoModel, AutoTokenizer
import numpy as np
import pickle

app = Flask(__name__)

# hard match model
dfa = DFA()

# softmatch model
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
pretrained_model = AutoModel.from_pretrained(r"ernie-3.0-medium-zh")
model = SentenceTransformer(pretrained_model)
params_path = "D:\code\qiji_compet\code\ir\softmatch\match_model\model_state.pdparams"
state_dict = paddle.load(params_path)
model.set_dict(state_dict)
# hnsw model
index_path = r"D:\code\qiji_compet\code\ir\softmatch\vector_index"
index = hnswlib.Index(space='cosine', dim=768)
index.load_index(index_path)
file_path = "D:\code\qiji_compet\code\data\dataset\multi_cls_data\dev_multi_v2.xlsx"
data_init = pd.read_excel(file_path)[:100]
sentences = data_init["content"].values.tolist()
labels = data_init["label"].values.tolist()
print("Loaded parameters from %s" % params_path)


# ===============hard match====================
@app.route('/hard_match/filter', methods=['POST', 'GET'])
def text_filter():
    string = request.args.get('word', '')
    response_dict = dict()
    if dfa.exists(string) is False:
        response_dict['is_illegal'] = False
    else:
        response_dict['is_illegal'] = True
        string = dfa.filter_all(string)
    response_dict['string'] = string
    response = json.dumps(response_dict)
    return response


@app.route('/hard_match/add', methods=['POST', 'GET'])
def add_new_words():
    string = request.args.get('word', '')
    if string in dfa.ban_words_set:
        return '"' + string + '"已在敏感词文档中，添加失败'
    dfa.add_new_word(string)
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
    if request.method == 'POST':
        text = request.data
        text = list(eval(text.decode("unicode_escape")).values())[0]  #lists
    else:
        text = request.args.get('contents')
    results = embedding(model, text, tokenizer)

    return results


@app.route('/soft_match/vector_update', methods=['POST', 'GET'])
def vector_update():
    path = request.args.get('file_path', '')
    data = pd.read_excel(path)[:100]
    sentences = data["content"].tolist()
    labels = data["label"].tolist()
    result = embedding(model, sentences, tokenizer, label=labels)
    content_bag = []
    for i, (con, lb) in enumerate(zip(sentences, labels)):
        content_bag.append({"label": lb, "vector": result[i]})
    content2embed = dict(zip(sentences, content_bag))  # {content1:{"label":,"vector":},content2:{}...}
    try:
        pickle.dump(content2embed, open(r"D:\code\qiji_compet\code\models\vector.pkl", "wb"))
        return {"embedding_result": "update vector successful"}
    except Exception as e:
        return {"embedding_result": str(e)}


@app.route('/soft_match/index_update', methods=['POST', 'GET'])
def Index_Update():
    vector_path = request.args.get('vector_path', '')
    retult = index_update(vector_path)

    return retult


@app.route('/soft_match/search', methods=['POST', 'GET'])
def Search():
    text = request.args.get('text', '')
    k = int(request.args.get('topk', ''))
    vector = embedding(model, text, tokenizer)
    print(len(list(vector.values())[0][0]))
    result = search(index, list(vector.values())[0][0], sentences, labels, k)
    return result


if __name__ == '__main__':
    port = 4567
    app.run('0.0.0.0', port)
