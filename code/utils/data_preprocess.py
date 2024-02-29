from string import punctuation as en_punctuation
import re
from zhon.hanzi import punctuation as cn_punctuation


class DataHelper:
    def __init__(self):
        punctuation = en_punctuation + cn_punctuation

    def check_string(self,s):
        pattern = r'.*?[\u4e00-\u9fa5a-zA-Z\d].'  # 匹配中英文及数字的正则表达式模式

        if re.match(pattern, s):
            return False
        else:
            return True

    def text_chunk(self, content: str) -> list:  # 切分
        content = content.lower()
        content = content.replace(" ", "")

        text_bag = re.split(r"[.。?？！]", content)
        print(text_bag)
        final_text_bag = []
        for text in text_bag:
            text = text.strip()
            if text == "":
                continue
            print(text, self.check_string(text))
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


if __name__ == '__main__':
    dh = DataHelper()
    result = dh.text_chunk(".'.'.'.'.'.'.'.你是是额呀，.'.'.....但是我不行哎")
    print(result)

