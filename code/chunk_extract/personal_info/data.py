# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.datasets import MapDataset


def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, "r", encoding="utf-8") as fin:
        for line in fin:
            key = line.strip("\n")
            vocab[key] = i
            i += 1
    return vocab


# 加载数据文件datafiles
def load_dataset(datafiles):
    # 读取数据文件data_path
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header  #Deleted by WGM
            # 处理每行数据（文本+‘\t’+标注）
            for line in fp.readlines():
                # 提取文本和标注
                words, labels = line.strip('\n').split('\t')
                # 文本中单字和标注构成的数组
                words = [k.strip() for k in words.split('\002')]
                labels = [k.strip() for k in labels.split('\002')]
                # 迭代返回文本和标注
                yield words, labels

    # 根据datafiles的数据类型，选择合适的处理方式
    if isinstance(datafiles, str):  # 字符串，单个文件名称
        # 返回单个文件对应的单个数据集
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):  # 列表或元组，多个文件名称
        # 返回多个文件对应的多个数据集
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


# 加载字典文件，文件由单列构成，需要设置value
def load_dict_single(dict_path):
    # 字典初始化为空
    vocab = {}
    # value是自增数值，从0开始
    i = 0
    # 逐行读取字典文件
    for line in open(dict_path, 'r', encoding='utf-8'):
        # 将每行文字设置为key
        line = line.strip('\n').split(" ")
        # 设置对应的value
        for label in line:
            label = label.strip()
            if label not in vocab:
                vocab[label] = i
                i += 1
    return vocab


def parse_decodes(sentences, predictions, lengths, label_vocab):
    """Parse the padding result

    Args:
        sentences (list): the tagging sentences.
        predictions (list): the prediction tags.
        lengths (list): the valid length of each sentence.
        label_vocab (dict): the label vocab.

    Returns:
        outputs (list): the formatted output.
    """
    predictions = [x for batch in predictions for x in batch]
    lengths = [x for batch in lengths for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lengths):
        sent = sentences[idx][:end]
        tags = [id_label[x] for x in predictions[idx][:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith("-B") or t == "O":
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split("-")[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append("".join([str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs
