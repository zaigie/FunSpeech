# -*- coding: utf-8 -*-
"""
SQLite数据库管理
"""

import sqlite3
import threading
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import os

from .config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite数据库管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.db_path = os.path.join(settings.DATA_DIR, "async_tts.db")
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地数据库连接"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_database(self):
        """初始化数据库表"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            conn = self._get_connection()
            cursor = conn.cursor()

            # 创建异步TTS任务表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS async_tts_tasks (
                    task_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    text TEXT NOT NULL,
                    voice TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL DEFAULT 16000,
                    format TEXT NOT NULL DEFAULT 'wav',
                    enable_subtitle BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_notify BOOLEAN NOT NULL DEFAULT FALSE,
                    notify_url TEXT,
                    audio_address TEXT,
                    sentences TEXT,
                    error_code INTEGER DEFAULT 20000000,
                    error_message TEXT DEFAULT 'RUNNING',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_status
                ON async_tts_tasks(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON async_tts_tasks(created_at)
            """)

            conn.commit()
            logger.info(f"数据库初始化完成: {self.db_path}")

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise

    def create_task(self, task_data: Dict[str, Any]) -> bool:
        """创建异步TTS任务"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO async_tts_tasks (
                    task_id, request_id, text, voice, sample_rate,
                    format, enable_subtitle, enable_notify, notify_url, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_data['task_id'],
                task_data['request_id'],
                task_data['text'],
                task_data['voice'],
                task_data['sample_rate'],
                task_data['format'],
                task_data['enable_subtitle'],
                task_data.get('enable_notify', False),
                task_data.get('notify_url'),
                'RUNNING',
                'RUNNING'
            ))

            conn.commit()
            logger.info(f"创建异步TTS任务: {task_data['task_id']}")
            return True

        except Exception as e:
            logger.error(f"创建任务失败: {e}")
            return False

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM async_tts_tasks WHERE task_id = ?
            """, (task_id,))

            row = cursor.fetchone()
            if row:
                task = dict(row)
                # 解析JSON字段
                if task['sentences']:
                    task['sentences'] = json.loads(task['sentences'])
                return task
            return None

        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return None

    def update_task_status(self, task_id: str, status: str,
                          audio_address: str = None,
                          sentences: List[Dict] = None,
                          error_code: int = 20000000,
                          error_message: str = "SUCCESS") -> bool:
        """更新任务状态"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            params = [status, error_code, error_message]
            sql_parts = ["status = ?", "error_code = ?", "error_message = ?", "updated_at = CURRENT_TIMESTAMP"]

            if audio_address:
                sql_parts.append("audio_address = ?")
                params.append(audio_address)

            if sentences:
                sql_parts.append("sentences = ?")
                params.append(json.dumps(sentences, ensure_ascii=False))

            if status in ['SUCCESS', 'FAILED']:
                sql_parts.append("completed_at = CURRENT_TIMESTAMP")

            params.append(task_id)

            cursor.execute(f"""
                UPDATE async_tts_tasks
                SET {', '.join(sql_parts)}
                WHERE task_id = ?
            """, params)

            conn.commit()
            logger.info(f"更新任务状态: {task_id} -> {status}")
            return True

        except Exception as e:
            logger.error(f"更新任务状态失败: {e}")
            return False

    def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取待处理的任务"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM async_tts_tasks
                WHERE status = 'RUNNING'
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"获取待处理任务失败: {e}")
            return []

    def cleanup_old_tasks(self, days: int = 7) -> int:
        """清理旧任务（默认7天前的任务）"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days)

            cursor.execute("""
                DELETE FROM async_tts_tasks
                WHERE created_at < ? AND status IN ('SUCCESS', 'FAILED')
            """, (cutoff_date,))

            deleted_count = cursor.rowcount
            conn.commit()

            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个旧任务")

            return deleted_count

        except Exception as e:
            logger.error(f"清理旧任务失败: {e}")
            return 0

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


# 全局数据库管理器实例
db_manager = DatabaseManager()