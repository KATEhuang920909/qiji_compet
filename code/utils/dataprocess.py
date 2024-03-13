from string import punctuation as en_punctuation
import re
from zhon.hanzi import punctuation as cn_punctuation
from collections import Counter


class DataPreprocess:
    def __init__(self):
        punctuation = en_punctuation + cn_punctuation

    def check_string(self, s):
        pattern = r'.*?[\u4e00-\u9fa5a-zA-Z\d].'  # 匹配中英文及数字的正则表达式模式

        if re.match(pattern, s):
            return False
        else:
            return True

    def remove_numbers(self, string):  # 删除数字
        new_string = ''
        for char in string:
            if not char.isnumeric():
                new_string += char
        return new_string

    def text_chunk(self, content: str) -> list:  # 切分
        content = content.lower()
        content = content.replace(" ", "")

        text_bag = re.split(r"[.。?？！]", content)
        final_text_bag = []
        for text in text_bag:
            text = text.strip()
            if text == "":
                continue
            if self.check_string(text):
                continue
            else:
                if len(text) > 128:
                    sub_text_bag = re.split(r"[,，：;:；]", text)
                    final_sub_text_bag = []
                    for t in sub_text_bag:
                        t = t.strip()
                        if t == "":
                            continue
                        if self.check_punctuation(t):
                            continue
                        else:
                            final_sub_text_bag.append(t)
                    final_text_bag += final_sub_text_bag
                else:
                    final_text_bag.append(text)
        return final_text_bag

    def chinese_check(self, text):
        zhmodel = re.compile(u'[\u4e00-\u9fa5]')  # 检查中文
        # zhmodel = re.compile(u'[^\u4e00-\u9fa5]')  #检查非中文
        match = zhmodel.search(text)
        if match:
            return True  # 包含中文
        else:
            return False


class DataPostprocess:
    def __init__(self):
        self.illegal_map = {"AD_Loan": "贷款类广告",
                            "AD_Network_service": "网络服务类广告",
                            "AD_Other": "广告",
                            "AD_Real_estate": "地产类广告",
                            "AD_Retail": "销售类广告",
                            "FR_Financial": "金融相关诈骗",
                            "FR_Phishing": "金融钓鱼诈骗",
                            "FR_Other": "网络诈骗",
                            "FUCK": "辱骂",
                            "SEX": "色情相关",
                            "IL_Escort_service": "色情相关",
                            "IL_Fake_ID_and_invoice": "代办、代理、代考等",
                            "IL_Gambling": "赌博相关",
                            "IL_Political_propaganda": "政治相关",
                            "POLITICAL": "政治相关",
                            "VIOLENT": "暴力、暴恐相关",
                            "FAKE": "代办、代理、代考等",
                            }
        self.private_map = {}

    def result_merge(self, soft_match_result):
        """


        :param soft_match_result: {"vec_search_result": final_result,"bm25_search_result": final_result}
        :return:
        """

        vec_label = [k[1] for k in soft_match_result["vec_search_result"] if k[2] <= 0.1]

        bm25_label = [k[1] for k in soft_match_result["bm25_search_result"]]

        label_counts = Counter(vec_label + bm25_label).items()
        sorted_counts = sorted(label_counts, key=lambda x: x[1], reverse=True)
        final_label = sorted_counts[0][0]
        return final_label

    def output_position_text(self, text: str, position: list, prepos=0) -> str:
        if position and position[0]:
            pos = position[0]
            pos[0] = pos[0] - prepos
            pos[1] = pos[1] - prepos
            print(pos)
            text = text[:pos[0]] \
                   + f":red[{text[pos[0]:pos[1]]}]" \
                   + self.output_position_text(text[pos[1]:], position[1:], len(text[:pos[0]]) + pos[1] - pos[0])
        else:
            return text
        return text

# if __name__ == '__main__':
# dp =DataPreprocess()
# dh=DataPostprocess()
# result = dp.text_chunk("1236")
# print(result)
# print(dh.output_position_text("行为你这种整的很我错，而且斯玛蒂第三军说的军事的v你", [[1, 3], [6, 8]]))
