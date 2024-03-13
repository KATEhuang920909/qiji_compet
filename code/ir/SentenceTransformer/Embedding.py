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
    return vector.numpy().tolist()


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
    import pickle
    import pandas as pd
    paddle.set_device(args.device)
    from MatchModel import SentenceTransformer
    # ErnieTinyTokenizer is special for ernie-tiny pretained model.
    params_path = "D:\work\qiji_compet\code\models\embedding_model\model_state.pdparams"
    tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-medium-zh")
    pretrained_model = ErnieModel.from_pretrained(r"ernie-3.0-medium-zh")
    embedding_model = SentenceTransformer(pretrained_model)
    state_dict = paddle.load(params_path)
    embedding_model.set_dict(state_dict)
    embedding_model.eval()

    data = pd.read_excel("knowledge_base.xlsx")
    sentences = data["content"].values.tolist()
    labels = data["label"].values.tolist()
    content_bag = []
    for i, (con, lb) in tqdm(enumerate(zip(sentences, labels))):
        result = embedding(embedding_model, con, tokenizer)
        content_bag.append({"label": lb, "vector": result[0]})
    content2embed = dict(zip(sentences, content_bag))  # {content1:{"label":,"vector":},content2:{}...}
    pickle.dump(content2embed, open(r"/vector.pkl", "wb"))
    print ("embedding_result", "update vector successful")

    #
    # while True:
    #     data1 = "我的易购我的优惠券中查询"
    #     data2 = "殊不知你爹我一个不小心用力过猛一巴掌将你婊子妈的狗头拍出脑震荡成了真正的白癡"
    #     data1 = input("1:")
    #     query_input_ids, query_token_type_ids = convert_example(data1, tokenizer, max_seq_length=args.max_seq_length)
    #     query_input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id)([query_input_ids])
    #     query_token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([query_token_type_ids])
    #     query_input_ids = paddle.to_tensor(query_input_ids)
    #     query_token_type_ids = paddle.to_tensor(query_token_type_ids)
    #     result1 = embedding_model.pooling(query_input_ids, query_token_type_ids)
    #     print(result1)
    #
    #     data2 = input("2:")
    #     query_input_ids, query_token_type_ids = convert_example(data2, tokenizer, max_seq_length=args.max_seq_length)
    #     query_input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id)([query_input_ids])
    #     query_token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([query_token_type_ids])
    #     query_input_ids = paddle.to_tensor(query_input_ids)
    #     query_token_type_ids = paddle.to_tensor(query_token_type_ids)
    #     result2 = embedding_model.pooling(query_input_ids, query_token_type_ids)
    #     print(result2)
    #     print(cosine_similarity(result1.numpy()[0].reshape(1, -1), result2.numpy()[0].reshape(1, -1)))
# FUCK 哈哈哈哈哈哈做纪念这帮逼还轻描淡写问你这办暂住证是要留着做纪念吗我纪念你ma逼啊我操
# 9976 0.0017091036 高仿真人民币代购qq FAKE
# 1542 0.001937449 PLACE移动尊敬的PLACE移动客户PLACE移动为您推出办流量包送最高DIGIT元红米手机一部活动多款手机任意选择无捆绑无最低消费详询移动营业厅 AD_Network_service
# 4592 0.0019578934 酱酱多多看给你急的买个股票别气得背过气你妈好不容易卖逼给你赚五十万还得给你看病本人从不介入股票涨跌与我无关可惜你没任何投资机会只能舔夏建统屁股走可悲吗学生党 FUCK
# 8063 0.002361834 当然爹知道你这废物在电脑旁瑟瑟发抖说爹复制粘贴然而废物东西跟ni智障爹一样怂de一b要不爹就不会给你智障亲爹带那么多次帽子 FUCK
# 4699 0.0023787618 PLACE百货大楼春季购物节暨婚庆节购物赢PLACENAMENAME款服饰折起会员抵分换购礼品NAME节化妆防晒节同步回td退 AD_Retail
# 5 0.0024462938 要注意身体不要太劳累 NORMAL
# 3095 0.0025324821 很想你了想见你 NORMAL
# 1477 0.0025632381 葛同学你睡了没啊 NORMAL
# 7285 0.0026184916 中午买了四个山竹比特么石头还硬撬不开啊馋死我了 NORMAL
# 8763 0.0026513934 手厅回馈老用户经典智能手机秒杀DIGIT元起只有DIGIT台每名用户限秒杀一台月日日每天点准时开抢快快登录神器手机营业厅错过就真没了客户端下载URL秒杀步骤手厅首页地区定位选择PLACE商品搜索秒杀参与秒杀PLACE联通 AD_Network_service

