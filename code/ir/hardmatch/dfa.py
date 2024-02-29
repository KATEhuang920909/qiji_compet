import os

current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.dirname(current_path)


class DFA:
    def __init__(self):
        self.ban_words_set = set()
        self.ban_words_list = list()
        self.ban_words_dict = dict()
        self.words_label = dict()  ############
        self.path = parent_path + r'/dicts/sensitive_words.txt'
        self.get_words()

    # 获取敏感词列表
    def get_words(self):
        with open(self.path, 'r', encoding='utf-8-sig') as f:
            for s in f:  # --------------add-----------------
                text = s.split("\t")
                label, word = text[1], text[-1].strip()
                if len(word) == 0:
                    continue
                else:
                    self.words_label[word] = label
                    if str(word) and word not in self.ban_words_set:
                        self.ban_words_set.add(word)
                        self.ban_words_list.append((str(word), label))
        self.add_hash_dict(self.ban_words_list)

    def change_words(self, path):
        self.ban_words_list.clear()
        self.ban_words_dict.clear()
        self.ban_words_set.clear()
        self.words_label.clear()  #########
        self.path = path
        self.get_words()

    # 将敏感词列表转换为DFA字典序
    def add_hash_dict(self, new_list):
        for x in new_list:
            self.add_new_word(x[0], x[1])

    # 添加单个敏感词
    def add_new_word(self, new_word, label):
        new_word = str(new_word)
        # print(new_word)
        now_dict = self.ban_words_dict
        i = 0
        for x in new_word:
            if x not in now_dict:
                x = str(x)
                new_dict = dict()
                new_dict['is_end'] = False
                now_dict[x] = new_dict
                now_dict = new_dict
            else:
                now_dict = now_dict[x]
            if i == len(new_word) - 1:
                now_dict['is_end'] = True
            i += 1
        self.words_label[new_word] = label

    # 寻找第一次出现敏感词的位置
    def find_illegal(self, _str):
        now_dict = self.ban_words_dict
        i = 0
        start_word = -1
        is_start = True  # 判断是否是一个敏感词的开始
        while i < len(_str):
            if _str[i] not in now_dict:
                if is_start is True:
                    i += 1
                    continue
                i = start_word + 1
                start_word = -1
                is_start = True
                now_dict = self.ban_words_dict
            else:
                if is_start is True:
                    start_word = i
                    is_start = False
                now_dict = now_dict[_str[i]]
                if now_dict['is_end'] is True:
                    labels = self.words_label[_str[start_word:i + 1]]
                    return start_word, i + 1, labels
                else:
                    i += 1
        return -1, None, None

    # 查找是否存在敏感词
    def exists(self, s):
        pos, _, _ = self.find_illegal(s)
        if pos == -1:
            return False
        else:
            return True

    # 将指定位置的敏感词替换为*
    def filter_words(self, filter_str, pos):
        flag = 0
        now_dict = self.ban_words_dict
        end_str = int()
        for i in range(pos, len(filter_str)):
            if filter_str[i] in now_dict and now_dict[filter_str[i]]['is_end'] is True:
                flag = 1
            else:
                if flag == 1:
                    end_str = i - 1
                    break
            now_dict = now_dict[filter_str[i]]
        num = end_str - pos + 1
        filter_str = filter_str[:pos] + '*' * num + filter_str[end_str + 1:]
        return filter_str

    def filter_all(self, s):
        result = []
        pos_list = list()
        ss = DFA.draw_words(s, pos_list)
        pos_label = self.find_illegal(ss)
        while pos_label[0] != -1:
            result.append(pos_label)
            ss = self.filter_words(ss, pos_label[0])
            pos_label = self.find_illegal(ss)
        # print(illegal_pos,ss)
        # i = 0
        # while i < len(ss):
        #     if ss[i] == '*':
        #         start = pos_list[i]
        #         while i < len(ss) and ss[i] == '*':
        #             i += 1
        #         i -= 1
        #         end = pos_list[i]
        #         num = end - start + 1
        #         s = s[:start] + '*' * num + s[end + 1:]
        #     i += 1
        return result

    @staticmethod
    def draw_words(_str, pos_list):
        ss = str()
        for i in range(len(_str)):
            if '\u4e00' <= _str[i] <= '\u9fa5' or '\u3400' <= _str[i] <= '\u4db5' or '\u0030' <= _str[i] <= '\u0039' \
                    or '\u0061' <= _str[i] <= '\u007a' or '\u0041' <= _str[i] <= '\u005a':
                ss += _str[i]
                pos_list.append(i)
        return ss


if __name__ == '__main__':
    with open("D:\work\QIJI\qiji_compet\code\dicts\sensitive_words.txt", 'r', encoding='utf-8-sig') as f:
        for s in f:
            text = s.split("\t")
            label, word = text[1], text[-1].strip()
