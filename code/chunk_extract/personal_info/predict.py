# -*- coding =  utf-8 -*-
"""
@Time : 2024/3/13 22:08
@Author: huangkai2
@File:predict.py
@Software: PyCharm
"""
import paddle
from paddlenlp.data import Pad, Stack
from data import parse_decodes
import re


class PersonalInfoCheck():
    def __init__(self):
        self.label_vocab = {'O': 0, 'B-prov': 1, 'I-prov': 2, 'E-prov': 3, 'B-city': 4, 'I-city': 5, 'E-city': 6,
                            'B-district': 7, 'I-district': 8, 'E-district': 9, 'B-town': 10, 'I-town': 11, 'E-town': 12,
                            'B-community': 46, 'I-community': 47, 'E-community': 48, 'B-poi': 16, 'E-poi': 18,
                            'B-road': 25, 'E-road': 27, 'B-roadno': 28, 'I-roadno': 29, 'E-roadno': 30, 'I-poi': 17,
                            'B-assist': 40, 'E-assist': 41, 'B-distance': 43, 'E-distance': 45, 'I-road': 26,
                            'B-intersection': 31, 'I-intersection': 32, 'E-intersection': 33, 'S-assist': 42,
                            'B-subpoi': 19, 'I-subpoi': 20, 'E-subpoi': 21, 'I-distance': 44, 'I-assist': 51,
                            'B-houseno': 22, 'I-houseno': 23, 'E-houseno': 24, 'B-cellno': 34, 'I-cellno': 35,
                            'E-cellno': 36, 'B-devzone': 13, 'I-devzone': 14, 'E-devzone': 15, 'B-floorno': 37,
                            'E-floorno': 39, 'I-floorno': 38, 'S-intersection': 52, 'B-village_group': 49,
                            'I-village_group': 53, 'E-village_group': 50, 'B-raodno': 54, 'E-raodno': 55}
        self.ID_REGEX18 = r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|10|11|12)(?:0[1-9]|[1-2]\d|30|31)\d{3}[\dXx]"
        self.ID_REGEX15 = r"[1-9]\d{7}(?:0\d|10|11|12)(?:0[1-9]|[1-2][\d]|30|31)\d{3}"
        self.BANKID_REGEX = r"[1-9]\d{9,29}"

    def convert_to_features(self, data, tokenizer):
        tokenized_input = tokenizer(data, return_length=True, is_split_into_words="token")
        input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int32")(
            [tokenized_input["input_ids"]])  # input_ids
        token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int32")(
            [tokenized_input["token_type_ids"]])  # token_type_ids
        seq_len = Stack(dtype="int64")([tokenized_input["seq_len"]])  # seq_len

        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        seq_len = paddle.to_tensor(seq_len)
        return input_ids, token_type_ids, seq_len

    def ner_predict(self, model, sentence: list, label_vocab, tokenizer):
        all_preds = []
        all_lens = []
        input_ids, seg_ids, lens = self.convert_to_features(sentence, tokenizer=tokenizer)

        preds = model(input_ids, seg_ids, lengths=lens)
        # Drop CLS prediction
        preds = [pred for pred in preds.numpy()]
        all_preds.append(preds)
        all_lens.append(lens)
        # sentences = [example[0] for example in ds.data]
        results = parse_decodes(sentence, all_preds, all_lens, label_vocab)
        return results

    def findID18(self, input_str: str) -> list:
        result = re.findall(self.ID_REGEX18, input_str)
        return result

    def findID15(self, input_str: str) -> list:
        result = re.findall(self.ID_REGEX15, input_str)
        return result

    def findBankID(self, input_str: str) -> list:
        result = re.findall(self.BANKID_REGEX, input_str)
        return result

# if __name__ == "__main__":
#     from paddlenlp.transformers import ErnieModel, ErnieTokenizer
#     from model import ErnieGRUCRF
#
#     paddle.set_device(args.device)
#     text = ["你在上海吗，我在武汉"]
#     tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-medium-zh")
#
#     ernie = ErnieModel.from_pretrained("ernie-3.0-medium-zh")
#
#     model = ErnieGRUCRF(ernie, 300, len(label_vocab), 100)
#     model.eval()
#     params_path = r"D:\work\qiji_compet\code\models\ner_model\model_27482.pdparams"
#     if params_path and os.path.isfile(params_path):
#         state_dict = paddle.load(params_path)
#         model.set_dict(state_dict)
#
#     preds = predict(model, text, label_vocab)
#     print(preds)
