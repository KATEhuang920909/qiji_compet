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

parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str,
                    default=r'D:\work\QiJi\qiji_compet\code\ir\softmatch\embedding_model\model_state.pdparams',
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


def embedding(model, content: str, tokenizer, batch_size=1):
    query_input_ids, query_token_type_ids = convert_example(content, tokenizer, max_seq_length=args.max_seq_length)
    query_input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id)([query_input_ids])
    query_token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([query_token_type_ids])
    query_input_ids = paddle.to_tensor(query_input_ids)
    query_token_type_ids = paddle.to_tensor(query_token_type_ids)
    vector = model.pooling(query_input_ids, query_token_type_ids)

    # return {"embedding_result": result}
    return vector


#
if __name__ == "__main__":
    import numpy as np
    import os
    import paddle
    import random


    def seed_paddle(seed=1024):
        seed = int(seed)

        random.seed(seed)  # 设置随机函数种子

        os.environ['PYTHONHASHSEED'] = str(seed)  # 设置python环境种子
        np.random.seed(seed)  # 设置np库种子
        paddle.seed(seed)  # 设置paddlepaddle随机种子


    seed_paddle(seed=1024)
    from paddlenlp.transformers import ErnieModel, ErnieTokenizer, LinearDecayWithWarmup
    import os
    from sklearn.metrics.pairwise import cosine_similarity

    paddle.set_device(args.device)

    # ErnieTinyTokenizer is special for ernie-tiny pretained model.
    tokenizer = ErnieTokenizer.from_pretrained("D:\work\QiJi\qiji_compet\code\ir\softmatch\embedding_model")

    model = ErnieModel.from_pretrained("D:\work\QiJi\qiji_compet\code\ir\softmatch\embedding_model")
    model.eval()

    while True:
        data1 = "我的易购我的优惠券中查询"
        data2 = "殊不知你爹我一个不小心用力过猛一巴掌将你婊子妈的狗头拍出脑震荡成了真正的白癡"
        data1 = input("1:")
        query_input_ids, query_token_type_ids = convert_example(data1, tokenizer, max_seq_length=args.max_seq_length)
        query_input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id)([query_input_ids])
        query_token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([query_token_type_ids])
        query_input_ids = paddle.to_tensor(query_input_ids)
        query_token_type_ids = paddle.to_tensor(query_token_type_ids)
        result1 = model(query_input_ids, query_token_type_ids)
        print(result1)

        data2 = input("2:")
        query_input_ids, query_token_type_ids = convert_example(data2, tokenizer, max_seq_length=args.max_seq_length)
        query_input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id)([query_input_ids])
        query_token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([query_token_type_ids])
        query_input_ids = paddle.to_tensor(query_input_ids)
        query_token_type_ids = paddle.to_tensor(query_token_type_ids)
        result2 = model(query_input_ids, query_token_type_ids)
        print(result2)
        print(cosine_similarity(result1[1].numpy(), result2[1].numpy()))
