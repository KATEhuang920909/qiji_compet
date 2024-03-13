import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import cv2
import sys
from utils.dataprocess import DataPreprocess, DataPostprocess
import requests
from collections import Counter

sys.path.append("./utils")
sys.path.append("./chunk_extract/negative_info")
sys.path.append("./chunk_extract/personal_info")
import numpy as np
from paddleocr import PaddleOCR
import json
from KeyBert import chunk_extract

ocr = PaddleOCR(use_angle_cls=True)
preprocess = DataPreprocess()
postprocess = DataPostprocess()


def bytes_to_numpy(image_bytes, channels='BGR'):
    """
    图片格式转换 bytes -> numpy
    args:
        image_bytes(str): 图片的字节流
        channels(str): 图片的格式 ['BGR'|'RGB']
    return(array):
        转换后的图片
    """
    _image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np = cv2.imdecode(_image_np, cv2.IMREAD_COLOR)
    if channels == 'BGR':
        return image_np
    elif channels == 'RGB':
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np


def merge_intervals(intervals):
    intervals.sort()  # 先按起始位置从小到大排序

    merged = []
    for interval in intervals:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)  # 如果当前区间与已有结果集最后一个区间不重叠，则直接添加到结果集
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])  # 否则更新已有结果集最后一个区间的右边界为两者之间较大值

    return merged


def RUN_SOP(text: str, strategy) -> dict:  #
    # 数据清洗
    # final_result = {}
    # if type(contents) is str:
    # text_bag = datahelper.text_chunk(contents)  # list
    # for i, text in enumerate(text_bag):
    #

    # return final_result
    # ner检测（隐私抽取）
    # url = f"http://127.0.0.1:4567/ner/person_info_check?contents={text}"
    # r = requests.get(url=url)
    # print(r.text)
    # ner_result_json = json.loads(r.text)  ## 待定
    # {'ner_result': ["('打倒中共共产党，打倒中共,这个法轮功万岁，妈卖批也。。。。', 'A1')"]}
    final_result = {"text": text,
                    "is_illegal": False,
                    "position": [[]],
                    "label": ""}

    if strategy == "ILLEGAL":
        # 硬匹配
        url = f"http://127.0.0.1:4567/hard_match/filter?contents={text}"
        r = requests.get(url=url)
        hard_match_result_json = json.loads(r.text)
        # {'is_illegal': True, 'position': [[0, 4, '反动'], [7, 11, '反动'], [13, 15, '暴恐']]}
        if hard_match_result_json['is_illegal'] is True:
            final_result = {"text": text,
                            "is_illegal": hard_match_result_json["is_illegal"],
                            "position": [k[:2] for k in hard_match_result_json["position"]],
                            "label": ",".join(set([k[-1] for k in hard_match_result_json["position"]]))}
        else:
            if (len(text) == 1) or text.isdigit():  # 挡板
                return final_result
            else:
                # 软匹配（向量检索）：
                topk = 10
                url = f"http://127.0.0.1:4567/soft_match/search?text={text}&topk={topk}"
                r = requests.get(url=url)
                soft_match_result_json = json.loads(r.text)
                final_label = postprocess.result_merge(soft_match_result_json)
                # {'search_result':
                # [['中午记得打个电话给快递员看看他什么时候送,然后你不在的话让他送门卫那...', 'NORMAL', 0.02320396900177002],
                # ['真的很容易', 'NORMAL', 0.02664703130722046],
                # ['我的心情上面都更新了那么多天了', 'NORMAL', 0.030143380165100098],
                # ['我晕 这机会都不要', 'NORMAL', 0.03533339500427246],
                # ['来没来给我回个信', 'NORMAL', 0.03790527582168579]]}

                # 片段(短语)抽取keybert
            if final_label == "NORMAL":
                return final_result
            else:
                orig = text.replace(" ", "").replace("。", " ").replace("，", " ").replace("？", "").strip()
                orig = orig.split(" ")
                chunk_result = chunk_extract(text, orig)
                if chunk_result[0][-1] < 0.9:
                    position = chunk_result[0][2]
                else:
                    chunk_result = [(x, y, z, score) for (x, y, z, score) in chunk_result if score > 0.9]
                    position = [k[2] for k in chunk_result]
                    position = merge_intervals(position)
                # [('今天天气还不错但是你妈死了', '死了', (11, 13), 0.9743359159128777),
                # ('今天天气还不错但是你妈死了', '你妈', (9, 11), 0.973183968951763),
                # ('今天天气还不错但是你妈死了', '妈死了', (10, 13), 0.9698428837171166),
                # ('今天天气还不错但是你妈死了', '你妈死了', (9, 13), 0.9670800426186588),
                # ('今天天气还不错但是你妈死了', '你妈死', (9, 12), 0.9640382451074878),
                # ('今天天气还不错但是你妈死了', '但是你妈', (8, 12), 0.9345639370863443)]
                final_result = {"text": "".join(orig),
                                "is_illegal": True,
                                "position": position,
                                "label": final_label}
    elif strategy == "PRIVATE":
        pass
        # url = f"http://127.0.0.1:4567/ner/person_info_check?contents={text}"
        # r = requests.get(url=url)
        # print(r.text)
        # ner_result_json = json.loads(r.text)  ## 待定
        # final_result = {"text": "".join(orig),
        #                 "is_illegal": True,
        #                 "position": position,
        #                 "label": final_label}
        # {'ner_result': ["('打倒中共共产党，打倒中共,这个法轮功万岁，妈卖批也。。。。', 'A1')"]}
    return final_result


def input_lines(contents: list, strategy):
    assert strategy in ["ILLEGAL", "PRIVATE"]
    illegal_flag, final_flag, illegal_labels = False, False, []
    for i, info_lines in enumerate(contents):
        lines = ""
        if type(info_lines) == list:
            for info in info_lines:
                if info:
                    line_result = RUN_SOP(str(info), strategy)
                    if line_result["is_illegal"]:
                        illegal_flag = True
                        final_position = line_result["position"]
                        final_text = line_result["text"]
                        final_label = line_result["label"]
                        print("final_text", final_text)
                        print("final_label", final_label)
                        print("final_label", final_position)
                        if type(final_position[0])!=list:
                            final_position=[final_position]
                        final_text = postprocess.output_position_text(final_text, final_position)
                        lines += final_text
                    else:
                        lines += str(info) + " "
                else:
                    lines += str(info) + " "

        elif type(info_lines) == str and info_lines != "None":
            # print(str(info_lines))
            line_result = RUN_SOP(str(info_lines), strategy)
            if line_result["is_illegal"]:
                illegal_flag = True
                final_flag = True
                final_position = line_result["position"]
                final_text = line_result["text"]
                final_label = line_result["label"]
                # print(final_text, final_position)
                if type(final_position[0]) != list:
                    final_position = [final_position]
                final_text = postprocess.output_position_text(final_text, final_position)
                lines += final_text
        if illegal_flag:
            out_text = lines + f"\t:red[  -->违规，类型为{final_label}]\n"
            illegal_flag = False
            illegal_labels.append(final_label)
            st.write(out_text)
    if final_flag:
        if strategy == "ILLEGAL":
            ill_string = "、".join(set(illegal_labels))
            st.caption("消息疑似包含{}违规信息，请核实。".format(ill_string))
        elif strategy == "PRIVATE":
            st.caption("消息疑似包含个人隐私，请核实。")
    else:
        st.caption("无敏感信息，检测通过")



# 设置全局属性
st.set_page_config(
    page_title='5G消息敏感信息监测系统demo',
    page_icon=' ',
    layout='wide'
)

# 正文
st.title('5G消息敏感信息监测系统demo')
table1, table2 = st.tabs(['个人隐私检测', '不良信息检测'])
with table1:
    tab1, tab2, tab3 = st.tabs(['text', 'document', 'image'])

    with tab1:
        '''
        ```markdown
        *规范个人言语行为，保护个人安全信息*
        ```
        '''
        contents = st.text_input('please input text:', key=0)
        if contents:
            st.write("数据预览：")
            st.write(contents[:100])
            content_lines = preprocess.text_chunk(contents)  # list
            input_lines(content_lines, strategy="PRIVATE")

    with tab2:
        '''
                ```markdown
                *规范个人言语行为，保护个人安全信息*
                ```
        '''
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "txt", "pdf"], key=1)

        if uploaded_file is not None:
            name = uploaded_file.name.split(".")[-1]
            try:
                if name in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file, dtype="str")
                    content_lines = df.values
                    st.write("数据预览：")
                    st.write(df.head(5))
                    if len(df) >= 1:
                        input_lines(content_lines, strategy="PRIVATE")

                elif name == "txt":
                    bytes_data = uploaded_file.getvalue().decode("utf8")
                    content_lines = bytes_data.split("\n")
                    if len(content_lines) == 1:
                        text = content_lines[0]
                        if text.strip():
                            st.write("数据预览：")
                            st.write(text[:100])
                        else:
                            st.caption("无有效文本信息")
                    elif len(content_lines) > 1:  # 多行
                        st.write("数据预览：")
                        st.write(pd.DataFrame(content_lines[:5]))
                        input_lines(content_lines, strategy="PRIVATE")
                elif name == "pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
                    if text.strip():
                        st.write("数据预览：")
                        st.write(text[:100])
                        content_lines = preprocess.text_chunk(contents)  # list
                        input_lines(content_lines, strategy="PRIVATE")
                    else:
                        st.caption("无有效文本信息")



            except Exception as e:
                st.write(e)
    with tab3:
        '''
                ```markdown
                *规范个人言语行为，保护个人安全信息*
                ```
        '''
        # 上传图片并展示
        uploaded_file = st.file_uploader("上传一张图片", type="jpg", key=2)

        if uploaded_file is not None:
            # 将传入的文件转为Opencv格式
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            # 展示图片
            st.image(opencv_image, channels="BGR")
            # 解析图片

            ocr_result = ocr.ocr(opencv_image, cls=True)
            ocr_result = '\n'.join([line[1][0] for line in ocr_result[0]])

            st.write("数据预览：")
            st.write(ocr_result[:100])
            content_lines = preprocess.text_chunk(ocr_result)  # list
            input_lines(content_lines, strategy="PRIVATE")
with table2:
    tab1, tab2, tab3 = st.tabs(['text', 'document', 'image'])

    with tab1:
        '''
                ```markdown
                *鉴别网络不良信息，提高网络防范意识*
                ```
        '''
        contents = st.text_input('please input text:', key=3)
        if contents:
            st.write("数据预览：")
            st.write(contents[:100])
            content_lines = preprocess.text_chunk(contents)  # list
            input_lines(content_lines, strategy="ILLEGAL")

    with tab2:
        '''
                ```markdown
                *鉴别网络不良信息，提高网络防范意识*
                ```
        '''
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "txt", "pdf"], key=4)
        print(uploaded_file)
        if uploaded_file is not None:
            name = uploaded_file.name.split(".")[-1]


            try:
                if name in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file, dtype="str")
                    content_lines = df.values.tolist()
                    st.write("数据预览：")
                    st.write(df.head(5))
                    if len(df) >= 1:
                        input_lines(content_lines, strategy="ILLEGAL")

                elif name == "txt":
                    bytes_data = uploaded_file.getvalue().decode("utf8")
                    content_lines = bytes_data.split("\n")
                    if len(content_lines) == 1:
                        text = content_lines[0]
                        if text.strip():
                            st.write("数据预览：")
                            st.write(text[:100])
                        else:
                            st.caption("无有效文本信息")
                    elif len(content_lines) > 1:  # 多行
                        st.write("数据预览：")
                        st.write(pd.DataFrame(content_lines[:5]))
                        input_lines(content_lines, strategy="ILLEGAL")
                elif name == "pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
                    if text.strip():
                        st.write("数据预览：")
                        st.write(text[:100])
                        content_lines = preprocess.text_chunk(contents)  # list
                        input_lines(content_lines, strategy="ILLEGAL")
                    else:
                        st.caption("无有效文本信息")



            except Exception as e:
                st.write(e)
    with tab3:
        '''
                ```markdown
                *鉴别网络不良信息，提高网络防范意识*
                ```
        '''
        # 上传图片并展示
        uploaded_file = st.file_uploader("上传一张图片", type="jpg", key=5)

        if uploaded_file is not None:
            # 将传入的文件转为Opencv格式
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            # 展示图片
            st.image(opencv_image, channels="BGR")
            # 解析图片

            ocr_result = ocr.ocr(opencv_image, cls=True)
            ocr_result = '\n'.join([line[1][0] for line in ocr_result[0]])

            st.write("数据预览：")
            st.write(ocr_result[:100])
            content_lines = preprocess.text_chunk(ocr_result)  # list
            input_lines(content_lines, strategy="ILLEGAL")
