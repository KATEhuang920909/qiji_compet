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
from paddlenlp.transformers import ErnieTokenizer, ErnieModel
from tqdm import tqdm
# from predict_bigru_crf import load_dict, convert_tokens_to_ids, Predictor, args
import pickle
import jieba
from ernie_gru_crf.model import ErnieGRUCRF
from ernie_gru_crf.data import label_vocab
app = Flask(__name__)
current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.dirname(current_path)
# hard match model
dfa = DFA()
search_model = SEARCH()
##### softmatch model
# bm25_model_path = parent_path + r"/models/search_model/bm25model.pickle"
# bm25_model = pickle.load(open(bm25_model_path, 'rb'))
# print("loaded bm25model")
params_path = parent_path + r"/models/embedding_model/model_state.pdparams"
tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-medium-zh")
pretrained_model = ErnieModel.from_pretrained(r"ernie-3.0-medium-zh")
embedding_model = SentenceTransformer(pretrained_model)
state_dict = paddle.load(params_path)
embedding_model.set_dict(state_dict)
embedding_model.eval()
print("loaded embedding model")


## ner model
ner_model = ErnieGRUCRF(pretrained_model, 300, len(label_vocab), 100)
params_path = parent_path + r"/models/ner_model/model_27482.pdparams"
state_dict = paddle.load(params_path)
ner_model.set_dict(state_dict)
ner_model.eval()



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
        f.writelines('/n' + string)
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
    results = []
    try:
        text = eval(text)
    except:
        pass
    assert type(text) in [list, str]
    if type(text) == list:

        for txt in text:
            result = embedding(embedding_model, txt, tokenizer)
            results += result
        return {"embedding_result": results}

    elif type(text) == str:
        result = embedding(embedding_model, text, tokenizer)
        return {"embedding_result": result}
    # print(results)


@app.route('/soft_match/vector_update', methods=['POST', 'GET'])
def vector_update():
    file_path = request.args.get('file_path', '')
    save_path = request.args.get('save_path', '')
    data = pd.read_excel(file_path)
    sentences = data["content"].values.tolist()
    labels = data["label"].values.tolist()
    content_bag = []
    for i, (con, lb) in tqdm(enumerate(zip(sentences, labels))):
        result = embedding(embedding_model, con, tokenizer)
        content_bag.append({"label": lb, "vector": result[0]})
    content2embed = dict(zip(sentences, content_bag))  # {content1:{"label":,"vector":},content2:{}...}
    try:
        pickle.dump(content2embed, open(save_path + r"/vector.pkl", "wb"))
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
    result = search_model.index_update(contents, labels, embeddings)

    # contents_cut = [jieba.lcut(x) for x in contents]
    # result2 = search_model.bm25_update(contents_cut)

    # result.update(result2)
    return result


@app.route('/soft_match/search', methods=['POST', 'GET'])
def Search():
    text = request.args.get('text', '')
    k = int(request.args.get('topk', ''))
    print(text)
    vector = embedding(embedding_model, text, tokenizer)
    text_cut = list(text)

    search_result = search_model.search(vector, text, text_cut, k)
    # print(search_result)
    return search_result


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
