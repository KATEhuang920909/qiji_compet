import re

ID_FIND = r'\d{17}[\dX]|^\d{15}'
ID_CHECK = r'^[1-9]\d{5}(18|19|20)?\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}(\d|[Xx])$'
BANKID_REGEX = r"[1-9]\d{9,29}"

import re


def parse_id_card(contents):
    # 身份证号提取
    result=[]
    identity_cards = re.findall(ID_FIND, contents)
    if identity_cards != []:
        # 身份证号码规则验证
        for ident in identity_cards:
            if re.match(ID_CHECK, ident):
                result.append(ident)
    return result
    # # 提取生日和性别
    # year = id_card[6:10]
    # month = id_card[10:12]
    # day = id_card[12:14]
    # sex_id = id_card[-2]
    #
    # # 性别
    # sex = 'F' if int(sex_id) % 2 == 0 else 'M'
    #
    # return f'{year}-{month}-{day}', sex, 'Adult' if int(year) in [19, 20] or int(year) > 1900 else 'Child'


def extract_bank_id_numbers(text):
    # 正则表达式模式
    pattern = r'(?<!\d)(?:\d{16}|\d{19})(?!\d)'
    phone_numbers = re.findall(pattern, text)
    return phone_numbers



def validate_bank_card_number(contents):
    result=[]
    extract_result=extract_bank_id_numbers(contents)
    print("extract_result",extract_result)
    if extract_result!=[]:
        for id_number in extract_result:
            if len(id_number) in [16, 19]:
                card_number = str(id_number)
                card_number = card_number.replace(' ', '')  # 移除空格

                if not card_number.isdigit():  # 判断是否只包含数字
                    return False

                # 从最后一位数字开始遍历
                for i in range(len(card_number) - 2, -1, -2):
                    digit = int(card_number[i])
                    digit *= 2  # 偶数位数字乘以2
                    if digit > 9:
                        digit = digit // 10 + digit % 10  # 两位数结果相加
                    card_number = card_number[:i] + str(digit) + card_number[i + 1:]

                # 计算总和
                total = sum(int(x) for x in card_number)
                if total % 10 == 0:
                    result.append(id_number)
    return result

text = "我的银行卡号是6217932180473316，密码是123456。请注意保密。"
card_numbers = validate_bank_card_number(text)
print(card_numbers)
# 示例
text = '我的身份证是420606199209094512'
print(parse_id_card(text))