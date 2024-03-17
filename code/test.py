import paddle


# x = paddle.ones(name='x', shape=[2, 3, 5], dtype='float32')  # create a tensor with shape=[2, 3, 5]
# y = paddle.unstack(x, axis=1)  # unstack with second axis, which results 3 tensors with shape=[2, 5]
# print(type(y[0].numpy()))

def merge_intervals(intervals):
    intervals.sort()  # 先按起始位置从小到大排序

    merged = []
    for interval in intervals:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)  # 如果当前区间与已有结果集最后一个区间不重叠，则直接添加到结果集
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])  # 否则更新已有结果集最后一个区间的右边界为两者之间较大值

    return merged


if __name__ == '__main__':
    from paddlenlp.transformers import ErnieTokenizer
    tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-medium-zh")
    ss=tokenizer.tokenize("053身份证号银行卡号地址武汉市江汉区江发路9号")
    print(ss,len(ss))
    # ls = [('今天天气还不错但是你妈死了', '妈死了', (9, 12), 0.973062359627934),
    #       ('今天天气还不错但是你妈死了', '死了', (10, 12), 0.9634797822302748),
    #       ('今天天气还不错但是你妈死了', '妈', (9, 10), 0.9618883827306477),
    #       ('今天天气还不错但是你妈死了', '你妈', (8, 10), 0.9605204208539946),
    #       ('今天天气还不错但是你妈死了', '你', (8, 9), 0.9578418130648267),
    #       ('今天天气还不错但是你妈死了', '死', (10, 11), 0.9577857423186806),
    #       ('今天天气还不错但是你妈死了', '但是你妈', (6, 10), 0.9573486640880502),
    #       ('今天天气还不错但是你妈死了', '妈死', (9, 11), 0.954388489656844),
    #       ('今天天气还不错但是你妈死了', '你妈死', (8, 11), 0.9500515085967826)]
    # pos=[list(k[2]) for k in ls]
    # print(merge_intervals(pos))
    # import re
    # ls={'BankCardInfo': [], 'IDCardInfo': [[(7, 25), '420621199209094512']], 'AddressInfo': [('42062119', 'poi')]}
    # txt="我的身份证号为420621199209094512，我的银行卡号为6217932180473316"
    # bankinfo =ls["BankCardInfo"]
    # pattern_bankinfo="|".join([k[1] for k in bankinfo])
    # txt = re.sub(pattern_bankinfo, "", txt, flags=re.IGNORECASE)
    # idinfo = ls["IDCardInfo"]
    # pattern_idinfo = "|".join([k[1] for k in idinfo])
    # txt = re.sub(pattern_idinfo, "", txt, flags=re.IGNORECASE)
    # print(txt)
