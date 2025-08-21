# -*- coding: utf-8 -*-
"""
数字转换工具模块
实现阿拉伯数字与中文数字之间的准确转换
专门为语音识别和合成优化
"""

import re
import logging
from typing import Union

logger = logging.getLogger(__name__)


class NumberConverter:
    """数字转换器 - 优化版实现"""

    # 数字映射
    DIGIT_TO_CHINESE = {
        0: "零",
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
    }

    CHINESE_TO_DIGIT = {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }

    # 单位映射
    UNITS = ["", "十", "百", "千", "万", "十万", "百万", "千万", "亿"]

    CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}

    @classmethod
    def arabic_to_chinese(cls, number: Union[int, float, str]) -> str:
        """
        阿拉伯数字转中文数字

        Args:
            number: 要转换的数字

        Returns:
            中文数字字符串
        """
        try:
            # 转换为字符串处理
            if isinstance(number, (int, float)):
                number_str = str(number)
            else:
                number_str = str(number).strip()

            if not number_str or number_str == "0":
                return "零"

            # 处理负数
            is_negative = number_str.startswith("-")
            if is_negative:
                number_str = number_str[1:]

            # 处理小数
            if "." in number_str:
                integer_part, decimal_part = number_str.split(".")
                result = cls._convert_integer_part(
                    int(integer_part) if integer_part else 0
                )
                result += "点"
                for digit in decimal_part:
                    result += cls.DIGIT_TO_CHINESE[int(digit)]
            else:
                result = cls._convert_integer_part(int(number_str))

            return ("负" + result) if is_negative else result

        except Exception as e:
            logger.warning(f"阿拉伯数字转中文失败: {number}, 错误: {str(e)}")
            return str(number)

    @classmethod
    def _convert_integer_part(cls, num: int) -> str:
        """转换整数部分"""
        if num == 0:
            return "零"

        if num < 0:
            return "负" + cls._convert_integer_part(-num)

        # 处理1-9
        if num < 10:
            return cls.DIGIT_TO_CHINESE[num]

        # 处理10-19
        if num < 20:
            if num == 10:
                return "十"
            else:
                return "十" + cls.DIGIT_TO_CHINESE[num - 10]

        # 处理20-99
        if num < 100:
            tens = num // 10
            ones = num % 10
            result = cls.DIGIT_TO_CHINESE[tens] + "十"
            if ones > 0:
                result += cls.DIGIT_TO_CHINESE[ones]
            return result

        # 处理100-999
        if num < 1000:
            hundreds = num // 100
            remainder = num % 100
            result = cls.DIGIT_TO_CHINESE[hundreds] + "百"
            if remainder > 0:
                if remainder < 10:
                    result += "零" + cls.DIGIT_TO_CHINESE[remainder]
                elif remainder >= 10 and remainder < 20:
                    # 特殊处理110-119
                    result += "一十"
                    if remainder > 10:
                        result += cls.DIGIT_TO_CHINESE[remainder - 10]
                else:
                    result += cls._convert_integer_part(remainder)
            return result

        # 处理1000-9999
        if num < 10000:
            thousands = num // 1000
            remainder = num % 1000
            result = cls.DIGIT_TO_CHINESE[thousands] + "千"
            if remainder > 0:
                if remainder < 100:
                    result += "零" + cls._convert_integer_part(remainder)
                else:
                    result += cls._convert_integer_part(remainder)
            return result

        # 处理万
        if num < 100000000:  # 小于1亿
            wan = num // 10000
            remainder = num % 10000
            result = cls._convert_integer_part(wan) + "万"
            if remainder > 0:
                if remainder < 1000:
                    result += "零" + cls._convert_integer_part(remainder)
                else:
                    result += cls._convert_integer_part(remainder)
            return result

        # 处理亿
        yi = num // 100000000
        remainder = num % 100000000
        result = cls._convert_integer_part(yi) + "亿"
        if remainder > 0:
            if remainder < 10000000:  # 小于千万
                result += "零" + cls._convert_integer_part(remainder)
            else:
                result += cls._convert_integer_part(remainder)
        return result

    @classmethod
    def chinese_to_arabic(cls, text: str) -> str:
        """
        中文数字转阿拉伯数字

        Args:
            text: 包含中文数字的文本

        Returns:
            转换后的文本
        """
        if not text:
            return text

        try:
            # 首先处理连续的单个中文数字（如电话号码）
            consecutive_pattern = (
                r"[零一二三四五六七八九](?:[零一二三四五六七八九]){2,}"
            )

            def replace_consecutive_digits(match):
                consecutive_digits = match.group(0)
                # 将连续的单个中文数字转换为阿拉伯数字
                result_digits = ""
                for char in consecutive_digits:
                    if char in cls.CHINESE_TO_DIGIT:
                        result_digits += str(cls.CHINESE_TO_DIGIT[char])
                return result_digits

            # 先处理连续数字
            text = re.sub(consecutive_pattern, replace_consecutive_digits, text)

            # 使用正则表达式匹配中文数字模式，包括负数
            pattern = (
                r"负?[零一二三四五六七八九十百千万亿]+(?:点[零一二三四五六七八九]+)?"
            )

            def replace_chinese_number(match):
                chinese_num = match.group(0)
                try:
                    # 检查是否为有效的负数表达
                    if chinese_num.startswith("负") and cls._is_valid_negative_number(
                        match, text
                    ):
                        return str(cls._parse_chinese_number(chinese_num))
                    elif not chinese_num.startswith("负"):
                        return str(cls._parse_chinese_number(chinese_num))
                    else:
                        # 如果"负"字不是表示负数，则只转换后面的数字部分
                        return "负" + str(cls._parse_chinese_number(chinese_num[1:]))
                except:
                    return chinese_num

            # 替换文本中的中文数字
            result = re.sub(pattern, replace_chinese_number, text)
            return result

        except Exception as e:
            logger.warning(f"中文数字转阿拉伯数字失败: {text}, 错误: {str(e)}")
            return text

    @classmethod
    def _is_valid_negative_number(cls, match, full_text: str) -> bool:
        """
        判断"负"字是否表示负数

        Args:
            match: 正则匹配对象
            full_text: 完整文本

        Returns:
            True 如果是有效的负数表达
        """
        matched_text = match.group(0)
        start_pos = match.start()

        # 检查是否是常见的非数字用法
        common_negative_words = [
            "负心",
            "负责",
            "负担",
            "负面",
            "负荷",
            "负债",
            "负载",
            "负重",
        ]

        # 检查前文是否包含这些词汇
        for word in common_negative_words:
            word_start = start_pos - len(word) + 1
            if word_start >= 0 and word_start < start_pos:
                if full_text[word_start : start_pos + len(word) - 1] == word:
                    return False

        # 检查前后文上下文，确定是否为数值表达
        # 如果"负"字前面有表示数值、温度、海拔等的词汇，很可能是负数
        value_indicators = [
            "温度",
            "度",
            "海拔",
            "高度",
            "深度",
            "余额",
            "账户",
            "金额",
            "数值",
            "值",
            "分数",
            "成绩",
        ]

        # 检查前文
        context_before = full_text[max(0, start_pos - 10) : start_pos]
        for indicator in value_indicators:
            if indicator in context_before:
                return True

        # 检查后文
        end_pos = match.end()
        context_after = full_text[end_pos : min(len(full_text), end_pos + 10)]
        for indicator in value_indicators:
            if indicator in context_after:
                return True

        # 如果"负"字后面直接跟着数字相关的字符，很可能是负数
        number_chars = set("零一二三四五六七八九十百千万亿点")
        if len(matched_text) > 1 and matched_text[1] in number_chars:
            # 进一步检查：如果是句子开头或前面是标点符号，更可能是负数
            if (
                start_pos == 0
                or full_text[start_pos - 1] in "，。！？：；、（）【】「」"
            ):
                return True
            # 如果前面是空格或数字相关词汇，也很可能是负数
            if start_pos > 0 and (
                full_text[start_pos - 1].isspace()
                or any(
                    indicator in full_text[max(0, start_pos - 5) : start_pos]
                    for indicator in ["是", "为", "达", "到", "从", "至", "等于"]
                )
            ):
                return True

        return True  # 默认认为是负数

    @classmethod
    def _parse_chinese_number(cls, chinese_text: str) -> float:
        """解析中文数字"""
        if not chinese_text:
            return 0

        # 处理负数
        is_negative = chinese_text.startswith("负")
        if is_negative:
            chinese_text = chinese_text[1:]  # 移除"负"字

        # 处理小数
        if "点" in chinese_text:
            integer_part, decimal_part = chinese_text.split("点", 1)
            integer_value = (
                cls._parse_chinese_integer(integer_part) if integer_part else 0
            )
            decimal_value = 0
            for i, char in enumerate(decimal_part):
                if char in cls.CHINESE_TO_DIGIT:
                    decimal_value += cls.CHINESE_TO_DIGIT[char] * (0.1 ** (i + 1))
            result = integer_value + decimal_value
        else:
            result = cls._parse_chinese_integer(chinese_text)

        return -result if is_negative else result

    @classmethod
    def _parse_chinese_integer(cls, chinese_text: str) -> int:
        """解析中文整数"""
        if not chinese_text:
            return 0

        # 处理特殊情况
        if chinese_text == "零":
            return 0

        # 初始化变量
        result = 0
        current = 0
        temp = 0  # 临时存储当前段的值

        # 逐个字符处理
        for char in chinese_text:
            if char in cls.CHINESE_TO_DIGIT:
                temp = cls.CHINESE_TO_DIGIT[char]
            elif char == "十":
                if temp == 0:  # 处理"十"开头的情况
                    temp = 1
                current += temp * 10
                temp = 0
            elif char == "百":
                if temp == 0:
                    temp = 1
                current += temp * 100
                temp = 0
            elif char == "千":
                if temp == 0:
                    temp = 1
                current += temp * 1000
                temp = 0
            elif char == "万":
                if temp > 0:
                    current += temp
                    temp = 0
                if current == 0:
                    current = 1
                result += current * 10000
                current = 0
            elif char == "亿":
                if temp > 0:
                    current += temp
                    temp = 0
                if current == 0:
                    current = 1
                result += current * 100000000
                current = 0

        # 处理剩余的数字
        if temp > 0:
            current += temp

        return result + current

    @classmethod
    def convert_text_numbers(cls, text: str, direction: str = "auto") -> str:
        """
        转换文本中的数字

        Args:
            text: 输入文本
            direction: 转换方向 ("arabic_to_chinese", "chinese_to_arabic", "auto")

        Returns:
            转换后的文本
        """
        if not text:
            return text

        if direction == "arabic_to_chinese":
            return cls._convert_arabic_to_chinese_in_text(text)
        elif direction == "chinese_to_arabic":
            return cls.chinese_to_arabic(text)
        else:  # auto
            return cls._auto_convert_numbers_in_text(text)

    @classmethod
    def _convert_arabic_to_chinese_in_text(cls, text: str) -> str:
        """将文本中的阿拉伯数字转换为中文数字"""

        def replace_number(match):
            number_str = match.group(0)
            try:
                return cls.arabic_to_chinese(number_str)
            except:
                return number_str

        # 匹配阿拉伯数字（包括小数）
        pattern = r"\d+(?:\.\d+)?"
        return re.sub(pattern, replace_number, text)

    @classmethod
    def _auto_convert_numbers_in_text(cls, text: str) -> str:
        """智能转换文本中的数字"""
        # 检测文本中是否包含中文数字
        chinese_number_pattern = r"[零一二三四五六七八九十百千万亿]"
        arabic_number_pattern = r"\d+"

        has_chinese_numbers = bool(re.search(chinese_number_pattern, text))
        has_arabic_numbers = bool(re.search(arabic_number_pattern, text))

        # 如果同时存在，优先转换为阿拉伯数字（ITN场景）
        if has_chinese_numbers:
            return cls.chinese_to_arabic(text)
        elif has_arabic_numbers:
            return cls._convert_arabic_to_chinese_in_text(text)
        else:
            return text


# 便捷函数
def arabic_to_chinese(number: Union[int, float, str]) -> str:
    """阿拉伯数字转中文数字"""
    return NumberConverter.arabic_to_chinese(number)


def chinese_to_arabic(text: str) -> str:
    """中文数字转阿拉伯数字"""
    return NumberConverter.chinese_to_arabic(text)


def convert_text_numbers(text: str, direction: str = "auto") -> str:
    """转换文本中的数字"""
    return NumberConverter.convert_text_numbers(text, direction)


def apply_itn_to_text(text: str) -> str:
    """
    对文本应用逆文本标准化（ITN）
    主要用于语音识别结果的后处理

    Args:
        text: 语音识别结果文本

    Returns:
        应用ITN后的文本
    """
    if not text:
        return text

    # 应用数字转换
    text = chinese_to_arabic(text)

    # 可以在这里添加其他ITN规则
    # 例如：标点符号标准化、缩写展开等

    return text
