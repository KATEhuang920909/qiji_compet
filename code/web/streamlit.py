import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import cv2
import sys
sys.path.append("../")
sys.path.append("../utils")
from commonFileExtractor import Extract_Text_From_File
import numpy as np


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


# 设置全局属性
st.set_page_config(
    page_title='我是标题',
    page_icon=' ',
    layout='wide'
)

# 正文
st.title('5G消息敏感信息监测系统demo')
tab1, tab2, tab3 = st.tabs(['text', 'document', 'image'])

with tab1:
    '''
    ```python
    import cv2
    image = cv2.imread('image.png')
    ```
    '''
    tweet_input = st.text_input('Tweet:')
    if tweet_input == "text":
        st.write("output", "0.99124:red[colors]", )
        st.caption("this is test")

with tab2:
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "txt", "pdf"])
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


        except Exception as e:
            st.write(e)
with tab3:
    # 上传图片并展示
    uploaded_file = st.file_uploader("上传一张图片", type="jpg")

    if uploaded_file is not None:
        # 将传入的文件转为Opencv格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # 展示图片
        st.image(opencv_image, channels="BGR")
        # 解析图片
        text = "".join(Extract_Text_From_File(opencv_image))
        st.write(text)
