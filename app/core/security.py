# -*- coding: utf-8 -*-
"""
安全相关功能
包含鉴权、token验证等安全功能
"""

from typing import Optional
from fastapi import Request
from .config import settings
from .exceptions import AuthenticationException


def mask_sensitive_data(
    data: str, mask_char: str = "*", keep_prefix: int = 4, keep_suffix: int = 4
) -> str:
    """遮盖敏感数据

    Args:
        data: 需要遮盖的数据
        mask_char: 遮盖字符
        keep_prefix: 保留前缀字符数
        keep_suffix: 保留后缀字符数

    Returns:
        遮盖后的数据
    """
    if not data or len(data) <= keep_prefix + keep_suffix:
        return data

    prefix = data[:keep_prefix]
    suffix = data[-keep_suffix:] if keep_suffix > 0 else ""
    mask_length = len(data) - keep_prefix - keep_suffix
    mask = mask_char * mask_length

    return f"{prefix}{mask}{suffix}"


def validate_token(token: str, expected_token: Optional[str] = None) -> bool:
    """验证访问令牌

    Args:
        token: 客户端提供的token
        expected_token: 期望的token值（从环境变量读取），如果为None则鉴权可选

    Returns:
        bool: 验证结果
    """
    # 如果没有配置期望的token，则鉴权是可选的
    if expected_token is None:
        return True

    # 如果配置了期望的token，则必须提供token
    if not token:
        return False

    # 简单的token格式验证（长度检查）
    if len(token) < 10:
        return False

    # 验证token是否匹配
    if token != expected_token:
        return False

    return True


def validate_xls_token(request: Request, task_id: str = "") -> str:
    """验证X-NLS-Token头部"""
    # 获取认证token
    token = request.headers.get("X-NLS-Token")

    # 如果没有配置XLS_TOKEN环境变量，则鉴权是可选的
    if settings.XLS_TOKEN is None:
        return token or "optional"

    # 如果配置了XLS_TOKEN，则必须提供token
    if not token:
        raise AuthenticationException("缺少访问令牌", task_id)

    if not validate_token(token, settings.XLS_TOKEN):
        masked_token = mask_sensitive_data(token)
        raise AuthenticationException(
            f"Gateway:ACCESS_DENIED:The token '{masked_token}' is invalid!", task_id
        )

    return token


def validate_bearer_token(request: Request, task_id: str = "") -> str:
    """验证Bearer Token鉴权（OpenAI兼容接口）"""
    # 获取Authorization头
    auth_header = request.headers.get("Authorization")

    # 如果没有配置XLS_TOKEN环境变量，则鉴权是可选的
    if settings.XLS_TOKEN is None:
        return auth_header or "optional"

    # 如果配置了XLS_TOKEN，则必须提供Authorization头
    if not auth_header:
        raise AuthenticationException("缺少Authorization头", task_id)

    # 检查Bearer格式
    if not auth_header.startswith("Bearer "):
        raise AuthenticationException(
            "Authorization头格式错误，应为'Bearer <token>'", task_id
        )

    # 提取token
    token = auth_header[7:]  # 去掉"Bearer "前缀

    if not validate_token(token, settings.XLS_TOKEN):
        masked_token = mask_sensitive_data(token)
        raise AuthenticationException(
            f"Gateway:ACCESS_DENIED:The token '{masked_token}' is invalid!", task_id
        )

    return token
