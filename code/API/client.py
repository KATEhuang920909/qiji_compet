# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     client
   Author :       huangkai
   date：          2024/2/24
-------------------------------------------------
"""
import json

import requests

# url = r"http://127.0.0.1:4567/soft_match/index_update?vector_path=D:\code\qiji_compet\code\models\vector.pkl"
# url = "http://127.0.0.1:4567/soft_match/vector_update?file_path=D:\code\qiji_compet\code\data\dataset\multi_cls_data\dev_multi_v2.xlsx"
# headers = {"Content-Type": "application/json"}
# url = r"http://127.0.0.1:4567/soft_match/index_update?vector_path=狗东西&topk=5"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)

# 获取向量库
# file_path = r"D:\work\qiji_compet\code\data\knowledge_data\knowledge_base.xlsx"
# save_path = r"D:\work\qiji_compet\code\models\search_model"
# url = fr"http://127.0.0.1:4567/soft_match/vector_update?file_path={file_path}&save_path={save_path}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)
# #
# 获取索引库

# save_path=r"D:\work\qiji_compet\code\models\search_model\vector.pkl"
# url = f"http://127.0.0.1:4567/soft_match/index_update?vector_path={save_path}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)

# embedding
# text = ['你是谁', '你', '啊']
# url = f"http://127.0.0.1:4567/soft_match/text2embedding?contents={text}"
# r = requests.get(url=url)
# result_json = json.loads(r.text)
# print(result_json)
# print(len(result_json["embedding_result"]))

# pre_url = "http://127.0.0.1:4567/soft_match/text2embedding?"
# # doc embedding
# content_bag = []
# txt="你是谁"
# doc_url = pre_url + f"contents={txt}"
# r = requests.get(url=doc_url)
# result_json = json.loads(r.text)

#
#
# text ="现在打开手机，关注微信公众号，扫码有惊喜"
# topk = 10
# text="中国共产党"
# url = f"http://127.0.0.1:4567/soft_match/search?text={text}&topk={topk}"
# r = requests.get(url=url)
# #
# result_json = json.loads(r.text)
# print(result_json)

# text ="你是谁qwertyuiop"
text ="你妈卖批哟，你是个大傻逼"
# embedding_type="sequence"
url = f"http://127.0.0.1:4567/hard_match/filter?contents={text}"
r = requests.get(url=url)
print(r.text)
result_json = json.loads(r.text)
print(result_json)


# text ="打倒中共共产党，打倒中共,这个法轮功万岁，妈卖批也。。。。"
# # embedding_type="sequence"
# url = f"http://127.0.0.1:4567/ner/person_info_check?contents={text}"
# r = requests.get(url=url)
# print(r.text)
# result_json = json.loads(r.text)
# print(result_json)
# {'search_result': [['NAME会员您好假期一定要旅yun行dong浮潜套装DIGIT起到店扫码更有惊喜URL回td退订', 'AD', -5.960464477539062e-07],
#                    ['都买回来，多备点哈', 'NORMAL', 2.956390380859375e-05],
#                    ['直播马上开始啦，我肿么那么高兴呢？我是多么热爱自己的工作啊！@盛博和我今天要和大家说说寒食十三绝，什么糖耳朵、螺丝转、墩饽饽、馓子、麻花、椰丝饼等等，全是我爱吃的，肿么办？肿么办？欢迎大家踊跃签到，说说你知道的点心吧！！！', 'NORMAL', 3.8743019104003906e-05], ['出来,我们在后门', 'NORMAL', 3.9458274841308594e-05], ['清明节的纪念，以后烧纸就对着它们吧！@car777999rielei@小兔0831', 'NORMAL', 4.1365623474121094e-05], ['在干吗？求物色位置。', 'NORMAL', 5.8650970458984375e-05], ['拇指会员专享快乐过暑期共赏儿童剧月日喜羊羊与灰太狼带您和孩子一起走进童话世界探寻童趣NAME抢亲子家庭套票大奖快回复', 'AD', 8.189678192138672e-05], ['PLACE地产花园路双地铁市中心唯一纯餐饮休闲街旺铺月交房即买即租即收益总裁亲批月享折钜NAME最后套正NAME湾PHONE', 'AD', 8.416175842285156e-05], ['PLACE集团寐minePLACE首家折扣店重装开业华美绽放全新商品折件再折日大NAMEa座楼专柜PHONE', 'AD', 8.934736251831055e-05], ['你好，我想知道哦东风的事', 'NORMAL', 8.994340896606445e-05]]}
