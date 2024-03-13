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
    data = open(dict_path).readlines()
    data = [k.strip().split(" ") for k in data]
    i = 0
    for label in data:
        for lb in label:
            if lb not in vocab:
                vocab[lb] = i
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
        tags = [id_label[x] for x in predictions[idx][:end]][1:]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.startswith("E"):
                if len(words):
                    words += s
                    sent_out.append(words)
                    tags_out.append(t.split("-")[1])
                words = ""
            elif t == "O":
                words = ""
                continue
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append([(s, t) for s, t in zip(sent_out, tags_out)])
    if len(sentences) == 1:
        return outputs[0]
    return outputs


if __name__ == '__main__':
    label_vocab = {'O': 0, 'B-prov': 1, 'I-prov': 2, 'E-prov': 3, 'B-city': 4, 'I-city': 5, 'E-city': 6,
                   'B-district': 7,
                   'I-district': 8, 'E-district': 9, 'B-town': 10, 'I-town': 11, 'E-town': 12, 'B-community': 46,
                   'I-community': 47, 'E-community': 48, 'B-poi': 16, 'E-poi': 18, 'B-road': 25, 'E-road': 27,
                   'B-roadno': 28, 'I-roadno': 29, 'E-roadno': 30, 'I-poi': 17, 'B-assist': 40, 'E-assist': 41,
                   'B-distance': 43, 'E-distance': 45, 'I-road': 26, 'B-intersection': 31, 'I-intersection': 32,
                   'E-intersection': 33, 'S-assist': 42, 'B-subpoi': 19, 'I-subpoi': 20, 'E-subpoi': 21,
                   'I-distance': 44,
                   'I-assist': 51, 'B-houseno': 22, 'I-houseno': 23, 'E-houseno': 24, 'B-cellno': 34, 'I-cellno': 35,
                   'E-cellno': 36, 'B-devzone': 13, 'I-devzone': 14, 'E-devzone': 15, 'B-floorno': 37, 'E-floorno': 39,
                   'I-floorno': 38, 'S-intersection': 52, 'B-village_group': 49, 'I-village_group': 53,
                   'E-village_group': 50,
                   'B-raodno': 54, 'E-raodno': 55}

    sentence = ["我的家在武汉市江岸区金融街25号"]
    all_preds = []
    all_lens = []
    preds = [[0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 25, 26, 27, 28, 30, 0]]
    all_preds.append(preds)
    all_lens.append([17])
    results = parse_decodes(sentence, all_preds, all_lens, label_vocab)
    print(results)
