import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import cv2
import sys
from utils.data_preprocess import DataHelper
import requests
sys.path.append("../")
sys.path.append("./utils")
import numpy as np
from paddleocr import PaddleOCR
import json
ocr = PaddleOCR(use_angle_cls=True)
datahelper = DataHelper()


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


def RUN_SOP(text: str, ):#
    # 数据清洗
    text_bag = datahelper.text_chunk(text)  #list
    for unit in text_bag:
        # 硬匹配
        url = f"http://127.0.0.1:4567/hard_match/filter?contents={unit}"
        r = requests.get(url=url)
        result_json = json.loads(r.text)
        if result_json['is_illegal']:
            return result_json['position']
        else:
            # ner检测（隐私抽取）



    # 向量匹配
    # 片段抽取（不良信息抽取）
    pass


# 设置全局属性
st.set_page_config(
    page_title='5G消息敏感信息监测系统demo',
    page_icon=' ',
    layout='wide'
)

# 正文
st.title('5G消息敏感信息监测系统demo')
table1, table2 = st.tabs(['待发送消息检测（可检出辱骂、涉政、涉黄、隐私信息）', '接收消息检测（可检出涉政、涉黄、广告、诈骗）'])
with table1:
    tab1, tab2, tab3 = st.tabs(['text', 'document', 'image'])

    with tab1:
        '''
        ```text
        Wise men say only fools rush in but I can't help falling in love with you
        ```
        '''
        text = st.text_input('please input text:', key=0)
        if text:

            print(type(text))
            """
            =======================
            填入文本消息预处理、检测、后处理方法
            
            =======================
            """

            st.write("output", "0.99124:red[colors]", )
            st.caption("this is test")

    with tab2:
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "txt", "pdf"], key=1)
        # if uploaded_file is not None:
        #     # To read file as bytes:
        #     bytes_data = uploaded_file.getvalue().decode("utf8")
        #     st.caption(bytes_data[:500] + "...")
        #     st.write(":red[" + bytes_data[300:400] + "]")

        if uploaded_file is not None:
            name = uploaded_file.name.split(".")[-1]
            try:
                if name in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file, dtype="str")
                    st.write("数据预览：")
                    st.write(df.head(5))
                elif name == "txt":
                    bytes_data = uploaded_file.getvalue().decode("utf8")
                    st.caption(bytes_data[:500] + "...")
                    st.write(":red[" + bytes_data[300:400] + "]")
                elif name == "pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
                    st.write(text)
                    st.write(":red[" + text[100:120] + "]")
                """
                =======================
                填入文本消息预处理、检测、后处理方法
    
                =======================
                """

            except Exception as e:
                st.write(e)
    with tab3:
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

            """
            =======================
            填入文本消息预处理、检测、后处理方法
    
            =======================
            """
            result = ocr_result
            st.write(result)
with table2:
    tab1, tab2, tab3 = st.tabs(['text', 'document', 'image'])

    with tab1:
        '''
        ```text
        Wise men say only fools rush in but I can't help falling in love with you
        ```
        '''
        text = st.text_input('please input text:', key=3)
        if text == "text":
            """
            =======================
            填入文本消息预处理、检测、后处理方法

            =======================
            """

            st.write("output", "0.99124:red[colors]", )
            st.caption("this is test")

    with tab2:
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "txt", "pdf"], key=4)
        # if uploaded_file is not None:
        #     # To read file as bytes:
        #     bytes_data = uploaded_file.getvalue().decode("utf8")
        #     st.caption(bytes_data[:500] + "...")
        #     st.write(":red[" + bytes_data[300:400] + "]")

        if uploaded_file is not None:
            name = uploaded_file.name.split(".")[-1]
            try:
                if name in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file, dtype="str")
                    st.write("数据预览：")
                    st.write(df.head(5))
                elif name == "txt":
                    bytes_data = uploaded_file.getvalue().decode("utf8")
                    st.caption(bytes_data[:500] + "...")
                    st.write(":red[" + bytes_data[300:400] + "]")
                elif name == "pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
                    st.write(text)
                    st.write(":red[" + text[100:120] + "]")
                """
                =======================
                填入文本消息预处理、检测、后处理方法

                =======================
                """

            except Exception as e:
                st.write(e)
    with tab3:
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

            """
            =======================
            填入文本消息预处理、检测、后处理方法

            =======================
            """
            result = ocr_result
            st.write(result)