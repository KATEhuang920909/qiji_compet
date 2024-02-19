"""
@Project ：illegal_context_recognition
@File ：ac.py
@IDE ：PyCharm
@Author ：wujx
"""
# import ahocorasick
#
#
# class AhocorasickNer:
#     """
#     AC自动机
#     """
#     def __init__(self):
#         self.actree = ahocorasick.Automaton()
#
#     def add_keywords_by_file(self, file):
#         words = []
#         with open(file, 'r', encoding='utf-8') as f:
#             for line in f.readlines():
#                 words.append(line.rstrip())
#
#         for flag, word in enumerate(words):
#             self.actree.add_word(word, (flag, word))
#         self.actree.make_automaton()
#
#     def add_keywords(self, words):
#         for flag, word in enumerate(words):
#             self.actree.add_word(word, (flag, word))
#         self.actree.make_automaton()
#
#     def match_results(self, sentence):
#         ner_results = []
#         # i的形式为(index1,(index2,word))
#         # index1: 提取后的结果在sentence中的末尾索引
#         # index2: 提取后的结果在self.actree中的索引
#         for i in self.actree.iter(sentence):
#             ner_results.append((i[0], i[1][1]))
#         return ner_results

class Node(object):
    """节点类"""
    def __init__(self, char=""):
        self.children = dict()
        self.char = char
        self.word = ""
        self.fail = None
        self.tail = None


class AC_Automata(object):
    """ac 自动机"""
    def __init__(self):
        self.root = Node()

    def build_tree(self, key_words):
        """构建字典树"""
        for word in key_words:
            temp_node = self.root
            for char in word:
                if char not in temp_node.children:
                    temp_node.children[char] = Node()
                temp_node = temp_node.children[char]
            temp_node.word = word

    def build_ac(self):
        """构建ac自动机"""
        queue_list = list()
        queue_list.insert(0, self.root)

        while len(queue_list) > 0:
            temp_node = queue_list.pop()
            for char, children in temp_node.children.items():
                if temp_node == self.root:
                    children.fail = self.root
                else:
                    children.fail = temp_node.fail.children.get(char) or self.root

                if children.fail:
                    children.tail = children.fail if children.fail.word else children.fail.tail

                queue_list.insert(0, children)

    def search(self, content):
        temp_node = self.root
        for index, char in enumerate(content):
            while temp_node and char not in temp_node.children:
                temp_node = temp_node.fail
            temp_node = temp_node.children.get(char) if temp_node is not None else self.root

            if temp_node.word:
                yield ((index+1 - len(temp_node.word), index+1), temp_node.word)

            tail_node = temp_node.tail
            while tail_node:
                yield ((index+1 - len(tail_node.word), index+1), tail_node.word)
                tail_node = tail_node.tail


if __name__ == '__main__':
    words=[]
    with open(r"data\dicts\dict1.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words.append(line.rstrip())
    ac = AC_Automata()
    ac.build_tree(words)
    ac.build_ac()
    import time
    string = ("如图，绿色部分代表相等。1号组中绿色部分长度为1，代表了以2号结尾的后缀最长能和长度为1的前缀匹配。"
              "2号组中可以知道以3号结尾的后缀最长能与长度为2的前缀匹配。也就是说，我们用i=2,j=1分别指向错位的字符串的上位置和下位置的前一个位置，"
              "如果i,j位置上的字符相等，那么就可以可以更新 next[i] = j并两个都增加1。如果某个位置不同，如4号组所示，那么一般地，"
              "我们需要将下方的串再错位一格。但是，由于下方的前3格和上方24格是相同的，再错位的话，就会造成我们一开始所提到的问题。"
              "但是，此时我们已经更新了24位置的next[i]，我们只需要将其与最长相同前缀对应起来就行了。如4号组第2、3所示，"
              "蓝色部分就是以3号位结尾的后缀的最长相同前缀，我们已经将它对应过来了。现在只需要递归地判断对应后相应位置的字符是否相同在进行操作就行了。"
              "其代码为while(j && mode[i]!=mode[j+1]) j = ne[j])果这个过程中一直都有mode[i]!=mode[j+1]，"
              "那么j最终会取到0而退出循环，因为我们约定了next[1] = next[0] = 0。这个位置上的next值就为0。a= asearch(string)for i in a:")
    for _ in range(4):
        string=string+string
    t1=time.time()
    a = ac.search(string)
    for i in a:
        print(i)
    t2=time.time()
    print(t2-t1)
    print(len(string))
    for word in words:
        if word in string:
            print(word)
    print(time.time()-t2)
