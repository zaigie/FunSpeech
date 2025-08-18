# -*- coding: utf-8 -*-
"""
日志配置模块
统一的日志配置和管理
"""

import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path
from .config import settings


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> None:
    """设置应用日志配置

    Args:
        level: 日志级别
        log_file: 日志文件路径
        format_string: 日志格式字符串
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
    """
    # 使用传入的参数或配置文件中的设置
    log_level = level or settings.LOG_LEVEL
    log_file_path = log_file or settings.LOG_FILE
    log_format = format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size = max_bytes or settings.LOG_MAX_BYTES
    backup_files = backup_count or settings.LOG_BACKUP_COUNT

    # 创建处理器列表
    handlers = [logging.StreamHandler(sys.stdout)]

    # 如果指定了日志文件路径，添加文件处理器
    if log_file_path:
        # 确保日志目录存在
        log_dir = Path(log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # 使用RotatingFileHandler实现日志轮转
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_files,
            encoding="utf-8",
        )
        handlers.append(file_handler)
    else:
        # 如果没有指定日志文件，使用默认路径
        default_log_path = Path("logs/funspeech.log")
        default_log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            default_log_path,
            maxBytes=max_file_size,
            backupCount=backup_files,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True,  # 强制重新配置
    )

    # 设置第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # 降低一些噪音大的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        配置好的日志记录器实例
    """
    return logging.getLogger(name)
