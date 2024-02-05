import streamlit as st
import base64
from pathlib import Path
import tempfile


def writer():
    file = st.file_uploader("选择待上传的PDF文件", type=['pdf'])
    if st.button("点击"):
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                fp = Path(tmp_file.name)
                fp.write_bytes(file.getvalue())
                with open(tmp_file.name, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" ' \
                              f'width="800" height="1000" type="application/pdf">'
                st.markdown(pdf_display, unsafe_allow_html=True)