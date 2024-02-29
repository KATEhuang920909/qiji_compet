from flask import Flask, request
import sys
import os

sys.path.append("../")
sys.path.append("../ir/hardmatch")
sys.path.append("../ir/SentenceTransformer")
sys.path.append("../ir/softmatch")
sys.path.append("../chunk_extract/personal_info")

current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.dirname(current_path)
print(current_path)
exit()
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

from paddlenlp.data import Pad, Stack, Tuple
from predict_bigru_crf import load_dict, convert_to_features, Predictor, args
import pickle
from functools import partial

app = Flask(__name__)

# hard match model
dfa = DFA()

##### softmatch model
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
pretrained_model = AutoModel.from_pretrained(r"ernie-3.0-medium-zh")
model = SentenceTransformer(pretrained_model)
params_path = parent_path + "\ir\softmatch\match_model\model_state.pdparams"
state_dict = paddle.load(params_path)
model.set_dict(state_dict)

## hnsw model
index_path = parent_path + r"\ir\softmatch\vector_index.bin"
index = hnswlib.Index(space='cosine', dim=768)
index.load_index(index_path)
file_path = parent_path + r"\data\dataset\multi_cls_data\train_multi_v2.xlsx"
data_init = pd.read_excel(file_path)[:100]
sentences = data_init["content"].values.tolist()
labels = data_init["label"].values.tolist()
print("Loaded parameters from %s" % params_path)

## chunk extract model
label_vocab = load_dict(parent_path + r"\chunk_extract\personal_info\data\tag.dic")
word_vocab = load_dict(parent_path + r"\chunk_extract\personal_info\data\word.dic")
predictor = Predictor(parent_path + r"\chunk_extract\personal_info\output",
                      args.device,
                      args.batch_size,
                      args.use_tensorrt,
                      args.precision,
                      args.enable_mkldnn,
                      args.benchmark,
                      args.save_log_path,
                      )


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
    # if request.method == 'POST':
    #     text = request.data
    #     text = list(eval(text.decode("unicode_escape")).values())[0]  # lists
    # else:
    text = request.args.get('contents')
    embedding_type = request.args.get('embedding_type')
    try:
        text = eval(text)
    except:
        pass
    results = embedding(model, text, tokenizer, embedding_type)
    print(results)
    return results


@app.route('/soft_match/vector_update', methods=['POST', 'GET'])
def vector_update():
    file_path = request.args.get('file_path', '')
    save_path = request.args.get('save_path', '')
    data = pd.read_excel(file_path)[:100]  # .sample(n=200)
    sentences = data["content"].tolist()
    labels = data["label"].tolist()
    result = embedding(model, sentences, tokenizer)["embedding_result"]
    content_bag = []
    for i, (con, lb) in enumerate(zip(sentences, labels)):
        content_bag.append({"label": lb, "vector": result[i]})
    content2embed = dict(zip(sentences, content_bag))  # {content1:{"label":,"vector":},content2:{}...}
    try:
        pickle.dump(content2embed, open(save_path + r"\vector.pkl", "wb"))
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
    print(text)
    vector = embedding(model, text, tokenizer)
    print(len(list(vector.values())[0][0]))
    result = search(index, list(vector.values())[0][0], sentences, labels, k)
    return result


# ===============soft match====================
@app.route('/ner/person_info_check', methods=['POST', 'GET'])
def ner_predict():
    # if request.method == 'POST':
    #     text = request.data
    #     text = list(eval(text.decode("unicode_escape")).values())[0]  # lists
    # else:
    text = request.args.get('contents')
    if text.strip():
        text=text.strip()
        token_id,len_token_id=convert_to_features(text,word_vocab)

    try:
        text = eval(text)
    except:
        pass
    results = embedding(model, text, tokenizer, embedding_type)
    print(results)
    return results


if __name__ == '__main__':
    port = 4567
    app.run('0.0.0.0', port)
