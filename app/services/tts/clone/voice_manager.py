#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音色管理器
基于CosyVoice2官方API实现零样本音色克隆和管理功能
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from app.utils.audio import validate_reference_audio

# 音色文件目录配置
VOICES_DIR = ROOT_DIR / "voices"
VOICE_REGISTRY_CONFIG = VOICES_DIR / "voice_registry.json"
SPK_DIR = VOICES_DIR / "spk"
SPKINFO_FILE = SPK_DIR / "spk2info.pt"

logger = logging.getLogger(__name__)


class VoiceManager:
    """音色管理器 - 基于CosyVoice2官方API"""

    def __init__(self, cosyvoice_instance=None):
        """
        初始化音色管理器

        Args:
            cosyvoice_instance: CosyVoice2实例，如果为None则延迟获取
        """
        # 确保目录存在
        VOICES_DIR.mkdir(exist_ok=True)
        SPK_DIR.mkdir(exist_ok=True)

        # 保存路径引用
        self.voices_dir = VOICES_DIR
        self.registry_file = VOICE_REGISTRY_CONFIG
        self.spk_dir = SPK_DIR
        self.spkinfo_file = SPKINFO_FILE
        self.cosyvoice = cosyvoice_instance

        # 加载或创建注册表
        self.registry = self._load_registry()

        # 如果传入了cosyvoice实例，立即设置自定义保存方法
        if self.cosyvoice:
            self._setup_custom_save_spkinfo()

    def _load_registry(self) -> Dict[str, Any]:
        """加载或创建音色注册表"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"注册表文件损坏，将创建新注册表: {e}")

        # 创建默认注册表
        return {
            "version": "2.0",
            "description": "CosyVoice2音色注册表",
            "created_at": "",
            "updated_at": "",
            "voices": {},
        }

    def _save_registry(self):
        """保存音色注册表"""
        import datetime

        self.registry["updated_at"] = datetime.datetime.now().isoformat()
        if not self.registry.get("created_at"):
            self.registry["created_at"] = self.registry["updated_at"]

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
        logger.info(f"音色注册表已保存到: {self.registry_file}")

    def _setup_custom_save_spkinfo(self):
        """设置自定义的save_spkinfo方法"""
        import types

        def custom_save_spkinfo(cosyvoice_self):
            """自定义的save_spkinfo方法，保存到voices/spk目录"""
            import torch

            try:
                torch.save(cosyvoice_self.frontend.spk2info, str(self.spkinfo_file))
                logger.info(f"spkinfo已保存到: {self.spkinfo_file}")
            except Exception as e:
                logger.error(f"保存spkinfo失败: {e}")
                raise

        # 使用monkey patching替换原方法
        self.cosyvoice.save_spkinfo = types.MethodType(
            custom_save_spkinfo, self.cosyvoice
        )

        # 同时加载已有的spkinfo（如果存在）
        self._load_existing_spkinfo()

    def _load_existing_spkinfo(self):
        """加载已有的spkinfo文件"""
        if self.spkinfo_file.exists():
            try:
                import torch

                existing_spkinfo = torch.load(
                    str(self.spkinfo_file), map_location="cpu"
                )

                # 合并到当前的spk2info中
                if hasattr(self.cosyvoice, "frontend") and hasattr(
                    self.cosyvoice.frontend, "spk2info"
                ):
                    self.cosyvoice.frontend.spk2info.update(existing_spkinfo)
                    logger.info(f"已加载 {len(existing_spkinfo)} 个保存的音色")

            except Exception as e:
                logger.warning(f"加载已保存的spkinfo失败: {e}")

    def _get_cosyvoice(self):
        """获取CosyVoice实例"""
        from app.core.config import settings

        model_mode = settings.TTS_MODEL_MODE.lower()
        if model_mode == "cosyvoice1":
            raise RuntimeError(
                "当前配置为仅使用SFT模型（TTS_MODEL_MODE=cosyvoice1），无法使用零样本音色克隆功能。"
                "如需使用零样本音色克隆，请设置环境变量 TTS_MODEL_MODE=all 或 TTS_MODEL_MODE=cosyvoice2"
            )
        if self.cosyvoice is None:
            from app.services.tts.engine import get_tts_engine

            # 使用统一的TTS引擎实例
            tts_engine = get_tts_engine()

            if not tts_engine.is_clone_model_loaded():
                raise RuntimeError("零样本克隆模型未加载，无法管理音色")

            self.cosyvoice = tts_engine.cosyvoice_clone

            # 设置自定义保存方法
            self._setup_custom_save_spkinfo()

        return self.cosyvoice

    def _find_voice_pairs(self) -> List[Tuple[str, Path, Path]]:
        """查找voices目录下的音色文件对"""
        pairs = []

        for txt_file in self.voices_dir.glob("*.txt"):
            wav_file = txt_file.with_suffix(".wav")
            if wav_file.exists():
                voice_name = txt_file.stem
                pairs.append((voice_name, txt_file, wav_file))
            else:
                logger.warning(f"找到txt文件但缺少对应的wav文件: {txt_file}")

        return pairs

    def _validate_and_prepare_audio(self, wav_file: Path) -> bool:
        """验证并准备音频文件"""
        # 验证音频文件
        is_valid, msg = validate_reference_audio(str(wav_file))
        if not is_valid:
            logger.warning(f"音频文件验证失败 {wav_file.name}: {msg}")

            # 尝试转换音频格式
            logger.info(f"尝试转换音频格式...")
            try:
                import torchaudio

                # 加载原始音频
                waveform, orig_sr = torchaudio.load(str(wav_file))

                # 转换为单声道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                    logger.info("已转换为单声道")

                # 重新保存为标准格式
                temp_wav_file = wav_file.parent / f"temp_{wav_file.name}"
                torchaudio.save(str(temp_wav_file), waveform, orig_sr)

                # 重新验证
                is_valid, msg = validate_reference_audio(str(temp_wav_file))
                if is_valid:
                    # 替换原文件
                    temp_wav_file.replace(wav_file)
                    logger.info("音频格式转换成功")
                    return True
                else:
                    temp_wav_file.unlink(missing_ok=True)
                    logger.error(f"转换后仍然无效: {msg}")
                    return False

            except Exception as e:
                logger.error(f"音频格式转换失败: {e}")
                return False

        return True

    def add_voice(self, voice_name: str, txt_file: Path, wav_file: Path) -> bool:
        """
        添加单个音色到模型中

        Args:
            voice_name: 音色名称
            txt_file: 文本文件路径
            wav_file: 音频文件路径

        Returns:
            bool: 是否成功添加
        """
        try:
            # 验证音频文件
            if not self._validate_and_prepare_audio(wav_file):
                return False

            # 读取参考文本
            with open(txt_file, "r", encoding="utf-8") as f:
                reference_text = f.read().strip()

            if not reference_text:
                logger.error(f"音色 {voice_name} 的文本文件为空")
                return False

            logger.info(f"正在添加音色: {voice_name}")
            logger.info(f"  音频文件: {wav_file}")
            logger.info(f"  参考文本: {reference_text}")

            # 获取CosyVoice实例
            cosyvoice = self._get_cosyvoice()

            # 加载音频
            from cosyvoice.utils.file_utils import load_wav
            import torchaudio

            # 检查音频文件基本信息
            try:
                audio_info = torchaudio.info(str(wav_file))
                audio_duration = audio_info.num_frames / audio_info.sample_rate
                logger.info(
                    f"  音频信息: 采样率={audio_info.sample_rate}, 长度={audio_duration:.2f}秒"
                )

                # 检查音频长度
                if audio_duration < 1.0:
                    logger.warning(
                        f"音频时长过短 ({audio_duration:.2f}秒)，建议使用3-30秒的音频"
                    )
                    return False
                elif audio_duration > 30.0:
                    logger.warning(
                        f"音频时长过长 ({audio_duration:.2f}秒)，建议控制在30秒以内以获得最佳效果"
                    )

            except Exception as e:
                logger.error(f"无法读取音频文件信息: {e}")
                return False

            # 加载音频数据 (16kHz)
            prompt_speech_16k = load_wav(str(wav_file), 16000)

            # 检查加载后的音频数据
            if prompt_speech_16k.shape[1] < 16000:  # 少于1秒
                logger.error(
                    f"音频数据太短: {prompt_speech_16k.shape[1]} 样本点 ({prompt_speech_16k.shape[1]/16000:.2f}秒)"
                )
                return False

            logger.info(f"  音频数据形状: {prompt_speech_16k.shape}")

            # 使用官方API添加音色
            success = cosyvoice.add_zero_shot_spk(
                reference_text, prompt_speech_16k, voice_name
            )

            if not success:
                logger.error(f"添加音色失败: {voice_name}")
                return False

            # 保存模型的spkinfo到自定义目录
            cosyvoice.save_spkinfo()
            logger.info(f"音色 {voice_name} 已保存到spkinfo文件中")

            # 更新注册表
            import datetime

            self.registry["voices"][voice_name] = {
                "name": voice_name,
                "reference_text": reference_text,
                "audio_file": str(wav_file.name),
                "text_file": str(txt_file.name),
                "file_size": os.path.getsize(wav_file),
                "audio_duration": audio_duration,
                "added_at": datetime.datetime.now().isoformat(),
                "status": "active",
            }

            logger.info(f"音色 {voice_name} 添加成功")
            return True

        except Exception as e:
            logger.error(f"添加音色失败 {voice_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def remove_voice(self, voice_name: str) -> bool:
        """
        从模型中移除音色

        Args:
            voice_name: 音色名称

        Returns:
            bool: 是否成功移除
        """
        try:
            # 获取CosyVoice实例
            cosyvoice = self._get_cosyvoice()

            # 检查音色是否存在
            if voice_name not in cosyvoice.frontend.spk2info:
                logger.warning(f"音色 {voice_name} 在模型中不存在")
                # 但仍然从注册表中移除
            else:
                # 从模型的spk2info中移除
                del cosyvoice.frontend.spk2info[voice_name]

                # 保存更新后的spkinfo
                cosyvoice.save_spkinfo()
                logger.info(f"音色 {voice_name} 已从模型中移除")

            # 从注册表中移除
            if voice_name in self.registry["voices"]:
                del self.registry["voices"][voice_name]
                self._save_registry()
                logger.info(f"音色 {voice_name} 已从注册表中移除")

            return True

        except Exception as e:
            logger.error(f"移除音色失败 {voice_name}: {str(e)}")
            return False

    def add_all_voices(self) -> Tuple[int, int]:
        """
        添加所有音色文件对

        Returns:
            Tuple[int, int]: (成功数量, 总数量)
        """
        logger.info("开始添加音色...")

        # 查找所有音色文件对
        voice_pairs = self._find_voice_pairs()

        if not voice_pairs:
            logger.info("未找到任何音色文件对 (*.txt + *.wav)")
            return 0, 0

        logger.info(f"找到 {len(voice_pairs)} 个音色文件对")

        success_count = 0
        total_count = len(voice_pairs)

        for voice_name, txt_file, wav_file in voice_pairs:
            # 检查是否已经存在
            cosyvoice = self._get_cosyvoice()
            if voice_name in cosyvoice.frontend.spk2info:
                logger.info(f"音色 {voice_name} 已存在，跳过")
                continue

            # 添加音色
            if self.add_voice(voice_name, txt_file, wav_file):
                success_count += 1

        # 保存注册表
        if success_count > 0:
            self._save_registry()
            logger.info(f"成功添加 {success_count}/{total_count} 个音色")
        else:
            logger.info("没有新的音色被添加")

        return success_count, total_count

    def list_voices(self) -> List[str]:
        """列出所有可用的音色"""
        try:
            cosyvoice = self._get_cosyvoice()
            # 从模型中获取所有音色，包括预训练和零样本克隆的
            all_voices = list(cosyvoice.frontend.spk2info.keys())
            return all_voices
        except Exception as e:
            logger.error(f"获取音色列表失败: {e}")
            # 回退到注册表
            return list(self.registry["voices"].keys())

    def list_clone_voices(self) -> List[str]:
        """列出所有零样本克隆音色"""
        return list(self.registry["voices"].keys())

    def get_voice_info(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """获取音色信息"""
        return self.registry["voices"].get(voice_name)

    def is_voice_available(self, voice_name: str) -> bool:
        """检查音色是否可用"""
        try:
            cosyvoice = self._get_cosyvoice()
            return voice_name in cosyvoice.frontend.spk2info
        except Exception:
            return False

    def refresh_voices(self) -> Tuple[int, int]:
        """刷新音色列表，重新扫描并添加新的音色文件"""
        logger.info("刷新音色列表...")
        return self.add_all_voices()

    def get_registry_info(self) -> Dict[str, Any]:
        """获取注册表信息"""
        clone_voices = self.list_clone_voices()
        total_voices = self.list_voices()

        return {
            "version": self.registry.get("version", "2.0"),
            "total_voices": len(total_voices),
            "clone_voices": len(clone_voices),
            "preset_voices": len(total_voices) - len(clone_voices),
            "created_at": self.registry.get("created_at", ""),
            "updated_at": self.registry.get("updated_at", ""),
            "voices": clone_voices,
        }


def main():
    """命令行工具入口"""
    import argparse

    parser = argparse.ArgumentParser(description="CosyVoice2音色管理工具")
    parser.add_argument("--add", action="store_true", help="添加所有音色文件对")
    parser.add_argument("--list", action="store_true", help="列出所有音色")
    parser.add_argument("--list-clone", action="store_true", help="列出零样本克隆音色")
    parser.add_argument("--remove", type=str, help="移除指定音色")
    parser.add_argument("--info", type=str, help="显示指定音色信息")
    parser.add_argument("--refresh", action="store_true", help="刷新音色列表")
    parser.add_argument("--registry-info", action="store_true", help="显示注册表信息")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    manager = VoiceManager()

    try:
        if args.add:
            success, total = manager.add_all_voices()
            print(f"添加完成: {success}/{total}")
        elif args.list:
            voices = manager.list_voices()
            if voices:
                print("所有可用音色:")
                for voice in voices:
                    print(f"  - {voice}")
            else:
                print("暂无可用音色")
        elif args.list_clone:
            voices = manager.list_clone_voices()
            if voices:
                print("零样本克隆音色列表:")
                for voice in voices:
                    print(f"  - {voice}")
            else:
                print("暂无零样本克隆音色")
        elif args.remove:
            if manager.remove_voice(args.remove):
                print(f"音色 {args.remove} 已成功移除")
            else:
                print(f"移除音色 {args.remove} 失败")
        elif args.info:
            info = manager.get_voice_info(args.info)
            if info:
                print(f"音色信息: {args.info}")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            else:
                print(f"音色 {args.info} 不存在")
        elif args.refresh:
            success, total = manager.refresh_voices()
            print(f"刷新完成: {success}/{total}")
        elif args.registry_info:
            info = manager.get_registry_info()
            print("注册表信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("请指定操作参数，使用 --help 查看帮助")

    except Exception as e:
        logger.error(f"操作失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
