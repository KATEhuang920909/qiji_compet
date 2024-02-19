"""
@File ：ac.py
@IDE ：PyCharm
@Author ：huangkai
"""


class FastTextConfig:
    def __init__(self):
        self.train_file = 'data/dataset/train.txt'
        self.test_file = 'data/dataset/test.txt'
        self.dev_file = 'data/dataset/dev.txt'

        self.model_path = 'outputs/fasttext/fasttext_model.bin'

        self.lr = 0.05
        self.embedding_dim = 400
        self.epoch = 10
        self.ngrams = 3
        self.loss_function = 'softmax'

        self.minCount = 1
        self.bucket = 20000


class SystemConfig:
    def __init__(self):
        self.illegal_dicts_file = 'dicts/illegal.txt'
        self.suspected_illegal_dicts_file = 'dicts/suspected_illegal.txt'
        self.trad2simple_file = 'dicts/trad2simp.txt'
        self.illegal_char_split_file = 'dicts/illegal_char_split.txt'

