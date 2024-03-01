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
    ls = [('今天天气还不错但是你妈死了', '妈死了', (9, 12), 0.973062359627934),
          ('今天天气还不错但是你妈死了', '死了', (10, 12), 0.9634797822302748),
          ('今天天气还不错但是你妈死了', '妈', (9, 10), 0.9618883827306477),
          ('今天天气还不错但是你妈死了', '你妈', (8, 10), 0.9605204208539946),
          ('今天天气还不错但是你妈死了', '你', (8, 9), 0.9578418130648267),
          ('今天天气还不错但是你妈死了', '死', (10, 11), 0.9577857423186806),
          ('今天天气还不错但是你妈死了', '但是你妈', (6, 10), 0.9573486640880502),
          ('今天天气还不错但是你妈死了', '妈死', (9, 11), 0.954388489656844),
          ('今天天气还不错但是你妈死了', '你妈死', (8, 11), 0.9500515085967826)]
    pos=[list(k[2]) for k in ls]
    print(merge_intervals(pos))
