#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字转换功能测试脚本
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.utils.number_converter import (
    NumberConverter,
    arabic_to_chinese,
    chinese_to_arabic,
    convert_text_numbers,
    apply_itn_to_text,
)


def test_arabic_to_chinese():
    """测试阿拉伯数字转中文数字"""
    print("=== 测试阿拉伯数字转中文数字 ===")

    test_cases = [
        (0, "零"),
        (1, "一"),
        (10, "十"),
        (11, "十一"),
        (20, "二十"),
        (21, "二十一"),
        (100, "一百"),
        (101, "一百零一"),
        (110, "一百一十"),
        (1000, "一千"),
        (10000, "一万"),
        (100000, "十万"),
        (1000000, "一百万"),
        (10000000, "一千万"),
        (100000000, "一亿"),
        (123456789, "一亿二千三百四十五万六千七百八十九"),
        (-123, "负一百二十三"),
        (3.14, "三点一四"),
        (-3.14, "负三点一四"),
    ]

    for input_num, expected in test_cases:
        result = arabic_to_chinese(input_num)
        status = "✓" if result == expected else "✗"
        print(f"{status} {input_num} -> {result} (期望: {expected})")


def test_chinese_to_arabic():
    """测试中文数字转阿拉伯数字"""
    print("\n=== 测试中文数字转阿拉伯数字 ===")

    test_cases = [
        ("零", "0"),
        ("一", "1"),
        ("十", "10"),
        ("十一", "11"),
        ("二十", "20"),
        ("二十一", "21"),
        ("一百", "100"),
        ("一百零一", "101"),
        ("一百一十", "110"),
        ("一千", "1000"),
        ("一万", "10000"),
        ("十万", "100000"),
        ("一百万", "1000000"),
        ("一千万", "10000000"),
        ("一亿", "100000000"),
        ("一亿二千三百四十五万六千七百八十九", "123456789"),
        ("负一百二十三", "-123"),
        ("三点一四", "3.14"),
        ("负三点一四", "-3.14"),
    ]

    for input_text, expected in test_cases:
        result = chinese_to_arabic(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} {input_text} -> {result} (期望: {expected})")


def test_complex_text_conversion():
    """测试复杂文本中的数字转换"""
    print("\n=== 测试复杂文本中的数字转换 ===")

    test_cases = [
        (
            "今天是一月十五日，温度是二十五度，花费了一百二十三元五角。",
            "今天是1月15日，温度是25度，花费了123元5角。",
        ),
        (
            "我有三个苹果，五个橙子，总共八个水果。",
            "我有3个苹果，5个橙子，总共8个水果。",
        ),
        (
            "这个项目需要三个月时间，预算是一万五千元。",
            "这个项目需要3个月时间，预算是15000元。",
        ),
        ("百分之五十的概率，千分之三的误差。", "100分之50的概率，1000分之3的误差。"),
        ("三点一四乘以二等于六点二八。", "3.14乘以2等于6.28。"),
    ]

    for input_text, expected in test_cases:
        result = chinese_to_arabic(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} 输入: {input_text}")
        print(f"   输出: {result}")
        print(f"   期望: {expected}")
        print()


def test_itn_function():
    """测试ITN功能"""
    print("\n=== 测试ITN功能 ===")

    test_cases = [
        (
            "语音识别结果：今天是一月十五日，温度二十五度。",
            "语音识别结果：今天是1月15日，温度25度。",
        ),
        ("用户说：我有三个苹果和五个橙子。", "用户说：我有3个苹果和5个橙子。"),
        (
            "识别文本：这个项目需要三个月，预算是一万五千元。",
            "识别文本：这个项目需要3个月，预算是15000元。",
        ),
    ]

    for input_text, expected in test_cases:
        result = apply_itn_to_text(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} 输入: {input_text}")
        print(f"   输出: {result}")
        print(f"   期望: {expected}")
        print()


def test_negative_numbers():
    """测试负数转换功能"""
    print("\n=== 测试负数转换功能 ===")

    test_cases = [
        # 基本负数测试
        ("负一", "-1"),
        ("负十", "-10"),
        ("负一百", "-100"),
        ("负一千", "-1000"),
        ("负三点一四", "-3.14"),
        ("负九点八", "-9.8"),
        # 复杂负数测试
        ("负一万二千三百", "-12300"),
        ("负三百四十五点六七", "-345.67"),
        # 上下文中的负数
        ("温度是负五度", "温度是-5度"),
        ("账户余额负一千元", "账户余额-1000元"),
        ("海拔负三百米", "海拔-300米"),
        # 混合正负数
        ("从负十度到二十度", "从-10度到20度"),
        ("负五加上十等于五", "-5加上10等于5"),
        # 非数字用法的"负"字（应该不被转换为负号）
        ("你这个负心汉欠我一千元", "你这个负心汉欠我1000元"),
        ("负责任的态度很重要", "负责任的态度很重要"),
        ("负担太重了有三十公斤", "负担太重了有30公斤"),
        ("负面情绪影响了五个人", "负面情绪影响了5个人"),
    ]

    for input_text, expected in test_cases:
        result = chinese_to_arabic(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} 输入: {input_text}")
        print(f"   输出: {result}")
        print(f"   期望: {expected}")
        print()


def test_auto_conversion():
    """测试智能转换功能"""
    print("\n=== 测试智能转换功能 ===")

    test_cases = [
        ("我有3个苹果和五个橙子。", "我有3个苹果和5个橙子。"),
        ("今天是一月十五日，温度25度。", "今天是1月15日，温度25度。"),
        ("这个项目需要三个月，预算一万五千元。", "这个项目需要3个月，预算15000元。"),
    ]

    for input_text, expected in test_cases:
        result = convert_text_numbers(input_text, "auto")
        status = "✓" if result == expected else "✗"
        print(f"{status} 输入: {input_text}")
        print(f"   输出: {result}")
        print(f"   期望: {expected}")
        print()


def test_number_converter_class():
    """测试NumberConverter类"""
    print("\n=== 测试NumberConverter类 ===")

    # 测试阿拉伯数字转中文
    converter = NumberConverter()
    result1 = converter.arabic_to_chinese(12345)
    print(f"12345 -> {result1}")

    # 测试中文转阿拉伯数字
    result2 = converter.chinese_to_arabic("一万二千三百四十五")
    print(f"一万二千三百四十五 -> {result2}")

    # 测试文本转换
    result3 = converter.convert_text_numbers(
        "我有三个苹果和五个橙子。", "chinese_to_arabic"
    )
    print(f"我有三个苹果和五个橙子。 -> {result3}")


def main():
    """主测试函数"""
    print("开始测试数字转换功能...\n")

    try:
        test_arabic_to_chinese()
        test_chinese_to_arabic()
        test_negative_numbers()
        test_complex_text_conversion()
        test_itn_function()
        test_auto_conversion()
        test_number_converter_class()

        print("\n所有测试完成！")

    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
