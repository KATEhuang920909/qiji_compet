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
import pickle
from paddlenlp.data import Pad, Tuple
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str,
                    default=r'D:\code\qiji_compet\code\models\match_model\model_state.pdparams',
                    help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu",
                    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


def convert_example(example, tokenizer, max_seq_length=512):
    query = example

    query_encoded_inputs = tokenizer(text=query, max_seq_len=max_seq_length)
    query_input_ids = query_encoded_inputs["input_ids"]
    query_token_type_ids = query_encoded_inputs["token_type_ids"]

    # title_encoded_inputs = tokenizer(text=title, max_seq_len=max_seq_length)
    # title_input_ids = title_encoded_inputs["input_ids"]
    # title_token_type_ids = title_encoded_inputs["token_type_ids"]

    return query_input_ids, query_token_type_ids  # , title_input_ids, title_token_type_ids


def embedding(model, content,  tokenizer,label=None, batch_size=1):
    examples = []
    if type(content) == str:
        query_input_ids, query_token_type_ids = convert_example(content, tokenizer, max_seq_length=args.max_seq_length)
        examples.append((query_input_ids, query_token_type_ids))
    elif type(content) in [list, numpy.ndarray]:
        for txt in tqdm(content):
            query_input_ids, query_token_type_ids = convert_example(txt, tokenizer, max_seq_length=args.max_seq_length)
            examples.append((query_input_ids, query_token_type_ids))
    # Separates data into some batches.
    batches = [examples[idx: idx + batch_size] for idx in range(0, len(examples), batch_size)]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
    ): [data for data in fn(samples)]
    result = np.zeros(shape=(1, 768))
    for batch in batches:
        query_input_ids, query_token_type_ids = batchify_fn(batch)

        query_input_ids = paddle.to_tensor(query_input_ids)
        query_token_type_ids = paddle.to_tensor(query_token_type_ids)
        vector = model.pooling(query_input_ids, query_token_type_ids=query_token_type_ids)

        result = np.concatenate((result, vector), axis=0)
    result = result.tolist()[1:]
    if type(content) == str:
        return {"embedding_result": result}
    elif type(content) in [list, numpy.ndarray]:
        content_bag = []
        for i, (con, lb) in enumerate(zip(content, label)):
            content_bag.append({"label": lb, "vector": result[i]})
        content2embed = dict(zip(content, content_bag))  # {content1:{"label":,"vector":},content2:{}...}
        try:
            pickle.dump(content2embed, open(r"D:\code\qiji_compet\code\models\vector.pkl", "wb"))
            return {"embedding_result": "update vector successful"}
        except Exception as e:
            return {"embedding_result": str(e)}
#
# if __name__ == "__main__":
#     paddle.set_device(args.device)
#
#     # ErnieTinyTokenizer is special for ernie-tiny pretained model.
#     tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
#
#     data = [
#         "世界上什么东西最小",
#         "光眼睛大就好看吗",
#         "小蝌蚪找妈妈怎么样",
#     ]
#
#     pretrained_model = AutoModel.from_pretrained(r"ernie-3.0-medium-zh")
#     model = SentenceTransformer(pretrained_model)
#
#     if args.params_path and os.path.isfile(args.params_path):
#         state_dict = paddle.load(args.params_path)
#         model.set_dict(state_dict)
#         print("Loaded parameters from %s" % args.params_path)
#     else:
#         raise ValueError("Please set --params_path with correct pretrained model file")
#
#     results = embedding(model, data, tokenizer, batch_size=args.batch_size)
#     for idx, text in enumerate(data):
#         print("Data: {} \t Embedding: {}".format(text, results[0][idx]))