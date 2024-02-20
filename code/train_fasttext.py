"""
@File ：ac.py
@IDE ：PyCharm
@Author ：huangkai
"""
from models.fasttext_model import FastText
from config import FastTextConfig

import random
def Train():
    # import pandas as pd
    # with open("data/dataset/train.txt", encoding="utf8") as f:
    #     train = f.readlines()
    # with open("data/dataset/dev.txt", encoding="utf8") as f:
    #     dev = f.readlines()
    # with open("data/dataset/test.txt", encoding="utf8") as f:
    #     test = f.readlines()
    # data = train + dev + test
    # random.shuffle(data)
    # for i in range(1,6):
    #     with open(f"data/dataset/dev{i}.txt", "w", encoding="utf8") as f:
    #         for unit in data[int((i-1)/5*len(data)):int(i/5*len(data))]:
    #             f.write(unit)
    #     with open(f"data/dataset/test{i}.txt", "w", encoding="utf8") as f:
    #         for unit in data[int((i-1)/5*len(data)):int(i/5*len(data))]:
    #             f.write(unit)
    #     with open(f"data/dataset/train{i}.txt", "w", encoding="utf8") as f:
    #         for unit in data[:int((i-1)/5*len(data))]+data[int(i/5*len(data)):]:
    #             f.write(unit)

        config = FastTextConfig()
        # config.train_file=f"data/dataset/train{i}.txt"
        # config.dev_file = f"data/dataset/dev{i}.txt"
        # config.test_file = f"data/dataset/test{i}.txt"
        classifier = FastText(config)
        model = classifier.train()

        classifier.save_mode(model)


        Test()


def Test():
    config = FastTextConfig()
    classifier = FastText(config, train=False)
    classifier.test()


if __name__ == '__main__':
    Train()

