import jieba
def create_ngram_model(text, n):
    words=jieba.lcut(text)
    ngrams = []  # 用于存储n-grams的列表

    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])  # 创建一个n-gram
        ngrams.append(ngram)

    return ngrams
text="你要不要脸啊？被包养很光荣啊？滚吧你@裴多酱强奸还拿钱，拿了30万行了。"
n = 2  # 选择2-gram模型
ngram_model = create_ngram_model(text, 3)

# 打印前10个2-grams
print(ngram_model)
