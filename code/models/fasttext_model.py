"""
@File ：ac.py
@IDE ：PyCharm
@Author ：huangkai
"""
import fasttext


class FastText:
    def __init__(self, config, train=True):
        self.model_path = config.model_path
        if train:
            self.classifier = fasttext
        else:
            self.classifier = fasttext.load_model(self.model_path)
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.dev_file = config.dev_file
        self.lr = config.lr
        self.dim = config.embedding_dim
        self.epoch = config.epoch
        self.word_ngrams = config.ngrams
        self.loss = config.loss_function
        self.minCount = config.minCount
        self.bucket = config.bucket

    def train(self):
        """

        :return:
        """
        model = self.classifier.train_supervised(self.train_file,
                                                 lr=self.lr,
                                                 dim=self.dim,
                                                 epoch=self.epoch,
                                                 word_ngrams=self.word_ngrams,
                                                 loss=self.loss,
                                                 minCount=self.minCount,
                                                 bucket=self.bucket)
        return model

    def test(self, model=None):
        """

        :return:
        """
        if model is None:
            model = self.classifier
        result = model.test(self.dev_file)
        print("测试样本数量：{}， 精确率：{}, 召回率：{}".format(result[0], result[1], result[2]))

    def predict(self, question):
        """

        :param question:str, 词之间用空格分开
        :return:
        """
        result = self.classifier.predict([question])
        return result[0][0][0]

    def save_mode(self, model):
        """

        :param model:
        :return:
        """
        model.save_model(self.model_path)


if __name__ == '__main__':
    with open("../data/dataset/test.txt", encoding="utf8") as f:
        test = f.readlines()
        test = [k.strip().split("\t") for k in test]
    model = fasttext.load_model("../outputs/fasttext/fasttext_model.bin")
    count = 0
    for unit in test:
        if unit[0] != model.predict(unit[1])[0][0]:
            print(unit[1])
            count += 1
    print(count,len(test))
