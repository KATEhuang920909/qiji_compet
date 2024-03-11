import os
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import pandas as pd
# from docx import Document
import time


def path_deal(path):
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower()
    return file_extension


def Extract_Text_From_File(image):
    """
    从不同类型的文件中提取文本。

    参数:
    - file_path: 文件的路径。

    返回:
    - 提取的文本字符串。
    """
    # 检查文件扩展名以决定处理方式

    # if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
    # 处理图片文件
    ocr = PaddleOCR(use_angle_cls=True)

    t1 = time.time()
    ocr_result = ocr.ocr(image, cls=True)
    print(time.time() - t1)
    return '\n'.join([line[1][0] for line in ocr_result[0]])

    # elif file_extension == '.pdf':
    #     # 处理PDF文件
    #     text = []
    #     with fitz.open(file_path) as pdf:
    #         for page_num in range(len(pdf)):
    #             page = pdf.load_page(page_num)
    #             text.append(page.get_text())
    #     return '\n'.join(text)

    # elif file_extension in ['.xls', '.xlsx']:
    #     # 处理Excel文件
    #     df = pd.read_excel(file_path, engine='openpyxl')
    #     return df.to_string()

    # elif file_extension == '.docx':
    #     # 处理Word文件
    #     doc = Document(file_path)
    #     return '\n'.join([para.text for para in doc.paragraphs])

    # elif file_extension == '.txt':
    #     # 处理文本文件
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         return file.read()

    # else:
    #     return "Unsupported file format."

# 使用示例
# 替换 'path/to/your/file' 为你的文件路径

# text = Extract_Text_From_File('../web/广告.jpg')
# print(text)
