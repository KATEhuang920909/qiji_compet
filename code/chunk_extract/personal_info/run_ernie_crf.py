# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddlenlp as ppnlp

import numpy as np
from functools import partial #partial()函数可以用来固定某些参数值，并返回一个新的callable对象
import pdb
import argparse
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default="./ernie_crf_ckpt", type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"], help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument("--data_dir", default="./waybill_ie/data", type=str, help="The folder where the dataset is located.")
args = parser.parse_args()
# fmt: on


def convert_example(example, tokenizer, label_vocab, max_seq_len=128, is_test=False):
    # 测试集没有标签
    if is_test:
        text = example
    else:
        text, label = example
    tokenizer_input = tokenizer.encode(text=text, max_seq_len=max_seq_len, pad_to_max_seq_len=False, return_length=True)
    input_ids = tokenizer_input["input_ids"]
    token_type_ids = tokenizer_input["token_type_ids"]
    seq_len = tokenizer_input["seq_len"]
    if not is_test:
        # 加入cls和sep
        label = ['O'] + label + ['O']
        # 将标签转为序列
        label = [label_vocab[x] for x in label]
        return input_ids, token_type_ids, seq_len, label
    else:  # 测试集，不返回标签
        return input_ids, token_type_ids, seq_len


@paddle.no_grad()
#评估函数
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()#评估器复位
    #依次处理每批数据
    for input_ids, seg_ids, lens, labels in data_loader:
        #CRF Loss
        preds = model(input_ids, seg_ids, lengths=lens)
        n_infer, n_label, n_correct = metric.compute(lens,preds,labels)
        metric.update(n_infer.numpy(),n_label.numpy(),n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("评估准确度: %.6f - 召回率: %.6f - f1得分: %.6f" % (precision, recall, f1_score))
    model.train()


# @paddle.no_grad()
# def predict(model, data_loader, ds, label_vocab):
#     all_preds = []
#     all_lens = []
#     for input_ids, seg_ids, lens, labels in data_loader:
#         preds = model(input_ids, seg_ids, lengths=lens)
#         # Drop CLS prediction
#         preds = [pred[1:] for pred in preds.numpy()]
#         all_preds.append(preds)
#         all_lens.append(lens)
#     sentences = [example[0] for example in ds.data]
#     results = parse_decodes(sentences, all_preds, all_lens, label_vocab)
#     return results


if __name__ == "__main__":
    import os
    from paddlenlp.data import Stack, Pad, Tuple
    from paddlenlp.metrics import ChunkEvaluator
    from model import ErnieGRUCRF
    from data import load_dataset,load_dict_single
    from SentenceTransformer.MatchModel import SentenceTransformer
    from paddlenlp.transformers import ErnieModel,ErnieTokenizer
    paddle.set_device(args.device)

    # Create dataset, tokenizer and dataloader.
    # 把数据集转为当个字符
    train_ds, dev_ds = load_dataset(datafiles=('./data/train.0', './data/dev.0'))

    # 把类别名转为id
    label_vocab = load_dict_single('./data/tag.txt')

    # 查看训练集和测试集的大小
    print("训练集大小:", len(train_ds))
    print("测试集集大小:", len(dev_ds))
    print(train_ds[0])
    print(dev_ds[0])
    print(label_vocab)  # 57个分类

    tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-medium-zh")
    # 偏函数，固定参数
    trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab, max_seq_len=128)

    # 对数据集进行编码（转为TinyBert需要的格式）
    train_ds.map(trans_func)
    dev_ds.map(trans_func)
    print(train_ds[0])

    # 数据组装成一个batch一个batch

    # 创建Tuple对象，将多个批处理函数的处理结果连接在一起
    ignore_label = -1
    # 因为数据集train_ds、dev_ds的每条数据包含4部分，所以Tuple对象中包含4个批处理函数
    batchify_fn = lambda samples, fn=Tuple(
        # 将每条数据的input_ids组合为数组，如果input_ids不等长，那么填充为pad_val
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        # 将每条数据的segment_ids组合为数组，如果segment_ids不等长，那么填充为pad_val
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        # 将每条数据的seq_len组合为数组
        Stack(),
        # 将每条数据的label组合为数组，如果label不等长，那么填充为pad_val
        Pad(axis=0, pad_val=ignore_label)
    ): fn(samples)

    # paddle.io.DataLoader加载给定数据集，返回迭代器，每次迭代访问batch_size条数据
    # 使用collate_fn定义所读取数据的格式
    # 训练集
    train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=32,
        return_list=True,
        collate_fn=batchify_fn)
    # 验证集
    dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=32,
        return_list=True,
        collate_fn=batchify_fn)

    pretrained_model = ErnieModel.from_pretrained(r"ernie-3.0-medium-zh")
    Model = SentenceTransformer(pretrained_model)
    params_path="D:\work\QiJi\qiji_compet\code\ir\softmatch\match_model\model_state.pdparams"
    state_dict = paddle.load(params_path)
    Model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
    ernie=Model.ptm
    model = ErnieGRUCRF(ernie, 300, len(label_vocab), 100)

    for name, param in model.Ernie.named_parameters():
        print(name, param.name, param.stop_gradient)

    # 设置Fine-Tune优化策略
    # 1.计算了块检测的精确率、召回率和F1-score。常用于序列标记任务，如命名实体识别
    metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
    # 2.在Adam的基础上加入了权重衰减的优化器，可以解决L2正则化失效问题
    optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())

    # 模型训练
    global_step = 0
    for epoch in range(20):
        # 依次处理每批数据
        for step, (input_ids, segment_ids, seq_lens, labels) in enumerate(train_loader, start=1):
            # 直接得到CRF Loss
            loss = model(input_ids, token_type_ids=segment_ids, lengths=seq_lens, labels=labels)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if global_step % 10 == 0:
                print("训练集的当前epoch:%d - step:%d" % (epoch, step))
                print("损失函数: %.6f" % (avg_loss))
            global_step += 1
        # 评估训练模型
        evaluate(model, metric, dev_loader)
        paddle.save(model.state_dict(),'./checkpoint/model_%d.pdparams' % (global_step))


    # preds = predict(model, test_loader, test_ds, label_vocab)
    # file_path = "ernie_crf_results.txt"
    # with open(file_path, "w", encoding="utf8") as fout:
    #     fout.write("\n".join(preds))
    # # Print some examples
    # print("The results have been saved in the file: %s, some examples are shown below: " % file_path)
    # print("\n".join(preds[:10]))
