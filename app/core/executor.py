# -*- coding: utf-8 -*-
"""异步执行器

把同步阻塞调用(主要是 HTTP 客户端 / 同步 WS session)派发到线程池,
避免阻塞 FastAPI 的事件循环。

线程数由 INFERENCE_THREAD_POOL_SIZE 控制, 默认 max(4, CPU 核数)。
"""

import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar
from functools import partial

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar("T")

# 全局线程池执行器
# 线程大部分时间阻塞在同步 WS recv() 等网络 I/O 上 (I/O-bound),
# 默认线程数从 cpu_count 调整为 cpu_count*8 (最低 32),
# 避免多并发 ASR 连接时线程池被耗尽导致服务不可用。
_DEFAULT_WORKERS = max(32, (os.cpu_count() or 4) * 8)
_MAX_WORKERS = int(os.getenv("INFERENCE_THREAD_POOL_SIZE", str(_DEFAULT_WORKERS)))

_executor: ThreadPoolExecutor = None


def get_executor() -> ThreadPoolExecutor:
    """获取全局线程池执行器（懒加载）"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS,
            thread_name_prefix="inference_worker"
        )
        logger.info(f"推理线程池已创建，最大工作线程数: {_MAX_WORKERS}")
    return _executor


def shutdown_executor():
    """关闭线程池执行器"""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("推理线程池已关闭")


async def run_sync(func: Callable[..., T], *args, **kwargs) -> T:
    """
    在线程池中执行同步函数，不阻塞事件循环

    Args:
        func: 同步函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        函数返回值

    Example:
        result = await run_sync(model.generate, input=audio_array, cache=cache)
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()

    # 使用 partial 绑定参数
    if kwargs:
        func_with_args = partial(func, *args, **kwargs)
    else:
        func_with_args = partial(func, *args) if args else func

    return await loop.run_in_executor(executor, func_with_args)
