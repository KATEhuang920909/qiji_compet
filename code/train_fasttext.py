"""
@File ：ac.py
@IDE ：PyCharm
@Author ：huangkai
"""
from models.fasttext_model import FastText
from config import FastTextConfig


def train():
    config = FastTextConfig()
    classifier = FastText(config)
    model = classifier.train()
    classifier.save_mode(model)


def test():
    config = FastTextConfig()
    classifier = FastText(config, train=False)
    classifier.test()


if __name__ == '__main__':
    train()
    test()
