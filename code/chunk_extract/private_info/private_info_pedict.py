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

class PrivateInfoCheck():
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
        self.ID_FIND = r'\d{17}[\dX]|^\d{15}'
        self.ID_CHECK = r'^[1-9]\d{5}(18|19|20)?\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}(\d|[Xx])$'
        self.EXTRACT_BANK_ID = r'(?<!\d)(?:\d{16}|\d{19})(?!\d)'

    def convert_to_features(self, sentence, tokenizer):
        tokenized_input = tokenizer(sentence, return_length=True, is_split_into_words="token")
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

    def extract_id_card_numbers(self, text):
        # 正则表达式模式
        position, numbers = [], []
        id_card_numbers = re.finditer(self.ID_FIND, text)
        for matchs in id_card_numbers:
            if matchs:
                position.append(matchs.span())
                numbers.append(matchs.group())
        return list(zip(position, numbers))

    def validate_id_card_number(self, text):
        result = []
        # 身份证号提取
        extract_result = self.extract_id_card_numbers(text)
        if extract_result != []:
            # 身份证号码规则验证
            for pos, id_num in extract_result:
                if re.match(self.ID_CHECK, id_num):
                    result.append([pos, id_num])
        return result
        # # 提取生日和性别
        # year = id_card[6:10]
        # month = id_card[10:12]
        # day = id_card[12:14]
        # sex_id = id_card[-2]
        #
        # # 性别
        # sex = 'F' if int(sex_id) % 2 == 0 else 'M'
        #
        # return f'{year}-{month}-{day}', sex, 'Adult' if int(year) in [19, 20] or int(year) > 1900 else 'Child'

    def extract_bank_card_numbers(self, text):
        # 正则表达式模式
        position, numbers = [], []
        bank_id_numbers = re.finditer(self.EXTRACT_BANK_ID, text)
        for matchs in bank_id_numbers:
            if matchs:
                position.append(matchs.span())
                numbers.append(matchs.group())
        return list(zip(position, numbers))

    def validate_bank_card_number(self, text):
        result = []
        extract_result = self.extract_bank_card_numbers(text)
        if extract_result != []:
            for pos, id_number in extract_result:
                if len(id_number) in [16, 19]:
                    card_number = str(id_number)
                    card_number = card_number.replace(' ', '')  # 移除空格

                    if not card_number.isdigit():  # 判断是否只包含数字
                        return False

                    # 从最后一位数字开始遍历
                    for i in range(len(card_number) - 2, -1, -2):
                        digit = int(card_number[i])
                        digit *= 2  # 偶数位数字乘以2
                        if digit > 9:
                            digit = digit // 10 + digit % 10  # 两位数结果相加
                        card_number = card_number[:i] + str(digit) + card_number[i + 1:]

                    # 计算总和
                    total = sum(int(x) for x in card_number)
                    if total % 10 == 0:
                        result.append([pos, id_number])
        return result

    # def private_info_check(self, model, txt: str, label_vocab, tokenizer):
    #     # 判断银行卡
    #     bankcard_result = self.validate_bank_card_number(txt)
    #     idcard_result = self.validate_id_card_number(txt)
    #     address_result = self.ner_predict(model, [txt], label_vocab, tokenizer)
    #     result = {"BankCardInfo": bankcard_result, # [pos,info]
    #               "IDCardInfo": idcard_result,
    #               "AddressInfo": address_result}
    #     return result
    def private_info_check(self, txt: str, ner_model, label_vocab, tokenizer):
        # 判断银行卡
        bankcard_result = self.validate_bank_card_number(txt)
        idcard_result = self.validate_id_card_number(txt)
        pattern_bankinfo = "|".join([k[1] for k in bankcard_result])
        ner_txt = re.sub(pattern_bankinfo, "", txt, flags=re.IGNORECASE)
        pattern_idinfo = "|".join([k[1] for k in idcard_result])
        ner_txt = re.sub(pattern_idinfo, "", ner_txt, flags=re.IGNORECASE)
        address_result = self.ner_predict(ner_model, [ner_txt], label_vocab, tokenizer)
        result = {"BankCardInfo": bankcard_result,  # [pos,info]
                  "IDCardInfo": idcard_result,
                  "AddressInfo": address_result}
        return result


# if __name__ == "__main__":
    from paddlenlp.transformers import ErnieModel, ErnieTokenizer
    from ner_model import ErnieGRUCRF
    import os
    # import re
    #
    # ls = ["武汉市", "金融港"]
    # regex = r'(?:{})'.format('|'.join(ls))
    # text = "我的家在武汉市金融港，金融大师"
    # res = re.finditer(regex, text)
    # for matchs in res:
    #     if matchs:
    #         print(matchs.span())
    # personalinfocheck = PrivateInfoCheck()
    # print(personalinfocheck.extract_bank_card_numbers("我的家在武汉市金融港"))
#     label_vocab=personalinfocheck.label_vocab
# #     paddle.set_device(args.device)
#     text = ["抵制共产党"]
#     tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-medium-zh")
#
#     ernie = ErnieModel.from_pretrained("ernie-3.0-medium-zh")
#
#     model = ErnieGRUCRF(ernie, 300, len(label_vocab), 100)
#
#     params_path = r"D:\work\qiji_compet\code\models\ner_model\model_27482.pdparams"
#     if params_path and os.path.isfile(params_path):
#         state_dict = paddle.load(params_path)
#         model.set_dict(state_dict)
#     model.eval()
#     preds = personalinfocheck.ner_predict(model, text, label_vocab, tokenizer)
#     print(preds)
#     text = '我的身份证是420606199209094512'
#     print(personalinfocheck.validate_id_card_number(text))


# text = "我的银行卡号是6217932180473316，密码是123456。请注意保密。"
# card_numbers = personalinfocheck.extract_bank_id_numbers(text)
# print(card_numbers,[text.index(card_numbers[0]),)
# # 示例


# #

# text = "我的手机号是6217932180473316和6215593202024518113."
# numbers =  personalinfocheck.validate_bank_card_number(text)
# print(numbers)


# # print(numbers)  # 输出: ['12345678901234567', '1234567890123456789']
