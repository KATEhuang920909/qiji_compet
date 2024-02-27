import re

ID_REGEX18 = r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|10|11|12)(?:0[1-9]|[1-2]\d|30|31)\d{3}[\dXx]"
ID_REGEX15 = r"[1-9]\d{7}(?:0\d|10|11|12)(?:0[1-9]|[1-2][\d]|30|31)\d{3}"
BANKID_REGEX = r"[1-9]\d{9,29}"

def findID18(input_str: str) -> list:
    result = re.findall(ID_REGEX18, input_str)
    return result


def findID15(input_str: str) -> list:
    result = re.findall(ID_REGEX15, input_str)
    return result


def findBankID(input_str: str) -> list:
    result = re.findall(BANKID_REGEX, input_str)
    return result


s = "我叫张三，我的身份证号码是992827200005063615，我喜欢学习、喜欢分享。我的银行卡号是992827200005063615"
print(findBankID(s))
