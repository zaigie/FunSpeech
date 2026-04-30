# -*- coding: utf-8 -*-
"""CosyVoice TTS 子服务

承载 CosyVoice2 / CosyVoice3 推理、零样本音色管理、文本规范化。
对外只输出原生采样率(24kHz / 22050Hz)的 raw 音频(WAV 容器),
格式 / 采样率 / 音量等转换由网关侧用 utils/audio 完成。

接口:
    GET    /health
    POST   /tts/file              整段离线合成
    WS     /tts/stream            流式合成
    GET    /voices                列出所有可用音色
    GET    /voices/{name}         单条音色信息
    POST   /voices                注册新音色 (multipart)
    DELETE /voices/{name}
    POST   /voices/refresh        扫描 voices/ 目录
    POST   /text/normalize        切句

启动:
    PORT=8004 uv run python server.py
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import os
import sys
import tempfile
import threading
import types
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response


logger = logging.getLogger("cosyvoice_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TTS_MODEL_MODE = os.getenv("TTS_MODEL_MODE", "all").lower()
TTS_DEVICE = os.getenv("TTS_DEVICE", "auto")
SFT_MODEL_ID = os.getenv("SFT_MODEL_ID", "iic/CosyVoice-300M-SFT")
CLONE_MODEL_ID = os.getenv("CLONE_MODEL_ID", "iic/CosyVoice2-0.5B")
COSYVOICE3_MODEL_ID = os.getenv(
    "COSYVOICE3_MODEL_ID", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
)
CLONE_MODEL_VERSION = os.getenv("CLONE_MODEL_VERSION", "cosyvoice3").lower()
TTS_LOAD_TRT = os.getenv("TTS_LOAD_TRT", "false").lower() == "true"
TTS_ENABLE_FP16 = os.getenv("TTS_ENABLE_FP16", "false").lower() == "true"
TTS_LOAD_VLLM = os.getenv("TTS_LOAD_VLLM", "false").lower() == "true"
MODELSCOPE_PATH = os.path.expanduser(
    os.getenv("MODELSCOPE_PATH", "~/.cache/modelscope/hub")
)
INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")

# 音色与注册表存储位置(由 docker-compose 通过卷 mount 进 /app/voices)
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
SPK_DIR = VOICES_DIR / "spk"
SPKINFO_FILE = SPK_DIR / "spk2info.pt"
REGISTRY_FILE = VOICES_DIR / "voice_registry.json"


def _ensure_voices_dirs() -> None:
    """运行期延迟创建,避免 import 期权限错误"""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    SPK_DIR.mkdir(parents=True, exist_ok=True)

PRESET_VOICES = ["中文女", "中文男", "日语男", "粤语女", "英文女", "英文男", "韩语女"]


# ---------------------------------------------------------------------------
# 把 submodule 加进 sys.path
# ---------------------------------------------------------------------------

THIRD_PARTY = Path(__file__).resolve().parent / "third_party" / "CosyVoice"
if THIRD_PARTY.exists():
    sys.path.insert(0, str(THIRD_PARTY))
    matcha = THIRD_PARTY / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))


# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------


def _detect_device(hint: str) -> str:
    if hint and hint not in ("auto", ""):
        return hint
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


DEVICE = _detect_device(TTS_DEVICE)


# ---------------------------------------------------------------------------
# 模型 + 音色管理(单例,服务全程持有)
# ---------------------------------------------------------------------------

_cosyvoice_sft = None
_cosyvoice_clone = None
_clone_model_actual_version = "cosyvoice2"  # 实际加载的版本
_load_lock = threading.Lock()
_registry: Dict[str, Any] = {}


def _load_registry_from_disk() -> Dict[str, Any]:
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("注册表损坏,重建: %s", exc)
    return {
        "version": "2.0",
        "description": "CosyVoice 子服务音色注册表",
        "created_at": "",
        "updated_at": "",
        "voices": {},
    }


def _save_registry() -> None:
    _ensure_voices_dirs()
    now = datetime.datetime.now().isoformat()
    _registry["updated_at"] = now
    if not _registry.get("created_at"):
        _registry["created_at"] = now
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(_registry, f, ensure_ascii=False, indent=2)


def _custom_save_spkinfo(self) -> None:
    """monkey-patch 进 cosyvoice 实例 — 把 spk2info.pt 存到 VOICES_DIR/spk/"""
    import torch

    _ensure_voices_dirs()
    torch.save(self.frontend.spk2info, str(SPKINFO_FILE))
    logger.info("spkinfo 已保存到: %s", SPKINFO_FILE)


def _load_existing_spkinfo(cosyvoice) -> None:
    """启动时把磁盘上已有的 spk2info.pt 合并进新加载的模型。"""
    if not SPKINFO_FILE.exists():
        return
    try:
        import torch

        existing = torch.load(str(SPKINFO_FILE), map_location="cpu")
        if hasattr(cosyvoice, "frontend") and hasattr(
            cosyvoice.frontend, "spk2info"
        ):
            cosyvoice.frontend.spk2info.update(existing)
            logger.info("已合并 %d 个保存的音色", len(existing))
    except Exception as exc:
        logger.warning("加载已存 spkinfo 失败: %s", exc)


def reload_voices_from_disk() -> Dict[str, Any]:
    """运行期热重载: 用磁盘上的 spk2info.pt + registry 覆盖内存状态。

    多副本部署时, 写副本 (cosyvoice-0) 修改音色后, 其它只读副本调用本函数
    即可立刻感知, 无需重启。

    与 _load_existing_spkinfo 不同, 这里是**替换**语义 — 磁盘上没有的
    音色会从内存里删掉, 保证 cosyvoice-0 的删除操作能传播到其它副本。
    """
    global _registry

    if _cosyvoice_clone is None:
        # 没加载 clone 模型, 没什么可重载的(预设音色不通过 spk2info 管理)
        _registry = _load_registry_from_disk()
        return {"clone_voices": 0, "registry_voices": 0, "clone_loaded": False}

    with _load_lock:
        # 1) 重新读取注册表
        _registry = _load_registry_from_disk()

        # 2) 重新读取 spk2info.pt, 替换内存 dict
        spk2info = getattr(_cosyvoice_clone.frontend, "spk2info", None)
        if spk2info is None:
            return {
                "clone_voices": 0,
                "registry_voices": len(_registry.get("voices", {})),
                "clone_loaded": False,
            }

        if SPKINFO_FILE.exists():
            try:
                import torch

                disk_state = torch.load(str(SPKINFO_FILE), map_location="cpu")
            except Exception as exc:
                logger.warning("reload 时加载 spkinfo 失败: %s", exc)
                disk_state = {}
        else:
            disk_state = {}

        # 替换语义: 删掉内存里磁盘没有的 zero-shot 音色, 但保留预设音色
        # (预设音色由模型自身在初始化时加载, 不在磁盘 spk2info.pt 内)
        registered = set(_registry.get("voices", {}).keys())
        for name in list(spk2info.keys()):
            if name in PRESET_VOICES:
                continue
            if name not in registered and name not in disk_state:
                del spk2info[name]

        # 把磁盘状态合并/覆盖进内存
        spk2info.update(disk_state)

        clone_count = len(_registry.get("voices", {}))
        logger.info(
            "音色已热重载: %d clone (来自 spk2info.pt), %d registry",
            len(disk_state),
            clone_count,
        )
        return {
            "clone_voices": len(disk_state),
            "registry_voices": clone_count,
            "clone_loaded": True,
        }


def _load_models() -> None:
    """根据 TTS_MODEL_MODE 加载模型"""
    global _cosyvoice_sft, _cosyvoice_clone, _clone_model_actual_version, _registry

    with _load_lock:
        if _cosyvoice_sft is not None or _cosyvoice_clone is not None:
            return

        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3  # type: ignore

        load_sft = TTS_MODEL_MODE in ("all", "sft")
        load_clone = TTS_MODEL_MODE in ("all", "clone")

        logger.info(
            "加载 CosyVoice (mode=%s, version=%s, device=%s, trt=%s, fp16=%s, vllm=%s)",
            TTS_MODEL_MODE,
            CLONE_MODEL_VERSION,
            DEVICE,
            TTS_LOAD_TRT,
            TTS_ENABLE_FP16,
            TTS_LOAD_VLLM,
        )

        if load_sft:
            try:
                _cosyvoice_sft = CosyVoice(
                    SFT_MODEL_ID,
                    load_jit=TTS_LOAD_TRT,
                    load_trt=TTS_LOAD_TRT,
                    fp16=TTS_ENABLE_FP16,
                    device=DEVICE,
                )
                logger.info("CosyVoice SFT 加载成功")
            except Exception as exc:
                logger.warning("SFT 加载失败: %s", exc)

        if load_clone:
            try:
                if CLONE_MODEL_VERSION == "cosyvoice3":
                    if TTS_ENABLE_FP16:
                        logger.warning(
                            "CosyVoice3 + FP16 + TRT 存在 NaN 风险, "
                            "建议 TTS_ENABLE_FP16=false"
                        )
                    _cosyvoice_clone = CosyVoice3(
                        COSYVOICE3_MODEL_ID,
                        load_trt=TTS_LOAD_TRT,
                        load_vllm=TTS_LOAD_VLLM,
                        fp16=TTS_ENABLE_FP16,
                        device=DEVICE,
                    )
                    _clone_model_actual_version = "cosyvoice3"
                else:
                    _cosyvoice_clone = CosyVoice2(
                        CLONE_MODEL_ID,
                        load_jit=TTS_LOAD_TRT,
                        load_trt=TTS_LOAD_TRT,
                        load_vllm=TTS_LOAD_VLLM,
                        fp16=TTS_ENABLE_FP16,
                        device=DEVICE,
                    )
                    _clone_model_actual_version = "cosyvoice2"
                logger.info(
                    "CosyVoice clone (%s) 加载成功",
                    _clone_model_actual_version,
                )

                # 安装 monkey-patch 的 save_spkinfo, 加载已存 spkinfo
                _cosyvoice_clone.save_spkinfo = types.MethodType(
                    _custom_save_spkinfo, _cosyvoice_clone
                )
                _load_existing_spkinfo(_cosyvoice_clone)
            except Exception as exc:
                logger.warning("clone 模型加载失败: %s", exc)

        if _cosyvoice_sft is None and _cosyvoice_clone is None:
            raise RuntimeError("CosyVoice 所有模型都加载失败")

        _registry = _load_registry_from_disk()


# ---------------------------------------------------------------------------
# 推理逻辑
# ---------------------------------------------------------------------------


def _format_prompt_for_clone(prompt_text: str) -> str:
    """根据 clone 模型版本拼接 instruct 文本前后缀。

    CosyVoice3 需要 'You are a helpful assistant.<...>' 前缀;
    CosyVoice2 仅需 '<|endofprompt|>' 后缀。
    """
    if _clone_model_actual_version == "cosyvoice3":
        if prompt_text and not prompt_text.startswith("You are"):
            return f"You are a helpful assistant. {prompt_text}<|endofprompt|>"
        if not prompt_text:
            return "You are a helpful assistant.<|endofprompt|>"
        if not prompt_text.endswith("<|endofprompt|>"):
            return f"{prompt_text}<|endofprompt|>"
        return prompt_text
    else:
        if prompt_text and not prompt_text.endswith("<|endofprompt|>"):
            return f"{prompt_text}<|endofprompt|>"
        return prompt_text


def _is_clone_voice(voice: str) -> bool:
    if _cosyvoice_clone is None:
        return False
    spk2info = getattr(_cosyvoice_clone.frontend, "spk2info", {})
    return voice in spk2info and voice in _registry.get("voices", {})


def _native_sample_rate(use_clone: bool) -> int:
    if use_clone:
        if _cosyvoice_clone is not None:
            return int(_cosyvoice_clone.sample_rate)
        return 24000
    if _cosyvoice_sft is not None:
        return int(_cosyvoice_sft.sample_rate)
    return 22050


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 单声道 -> WAV bytes"""
    buf = io.BytesIO()
    if audio.ndim == 2:
        audio = audio[0]
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def synthesize_offline(
    text: str,
    voice: str,
    speed: float,
    prompt: str,
    return_timestamps: bool,
) -> Tuple[bytes, int, Optional[List[Dict[str, Any]]]]:
    """整段离线合成. 返回 (wav_bytes, native_sr, sentences_or_None)"""

    use_clone = _is_clone_voice(voice)
    sentences_info: Optional[List[Dict[str, Any]]] = None

    if use_clone:
        if _cosyvoice_clone is None:
            raise RuntimeError("clone 模型未加载, 但请求需要 clone 音色")
        engine = _cosyvoice_clone
        formatted_prompt = _format_prompt_for_clone(prompt)
    else:
        if _cosyvoice_sft is None:
            raise RuntimeError("SFT 模型未加载, 但请求使用预设音色")
        engine = _cosyvoice_sft
        formatted_prompt = ""

    native_sr = int(engine.sample_rate)
    all_segments: List[np.ndarray] = []

    if return_timestamps:
        sentences_info = []
        current_time_ms = 0.0
        normalized = engine.frontend.text_normalize(
            text, split=True, text_frontend=True
        )
        for sentence in normalized:
            seg_chunks: List[np.ndarray] = []
            if use_clone:
                if prompt:
                    gen = engine.inference_instruct2(
                        sentence,
                        formatted_prompt,
                        None,
                        zero_shot_spk_id=voice,
                        stream=False,
                        speed=speed,
                    )
                else:
                    gen = engine.inference_zero_shot(
                        sentence,
                        "",
                        None,
                        zero_shot_spk_id=voice,
                        stream=False,
                        speed=speed,
                    )
            else:
                gen = engine.inference_sft(
                    sentence, voice, stream=False, speed=speed
                )

            for audio_data in gen:
                seg_chunks.append(audio_data["tts_speech"].numpy())

            if not seg_chunks:
                continue
            sentence_audio = (
                np.concatenate(seg_chunks, axis=1)
                if len(seg_chunks) > 1
                else seg_chunks[0]
            )
            duration_ms = sentence_audio.shape[1] / native_sr * 1000
            sentences_info.append(
                {
                    "text": sentence,
                    "begin_time": str(int(current_time_ms)),
                    "end_time": str(int(current_time_ms + duration_ms)),
                }
            )
            all_segments.append(sentence_audio)
            current_time_ms += duration_ms
    else:
        if use_clone:
            if prompt:
                gen = engine.inference_instruct2(
                    text,
                    formatted_prompt,
                    None,
                    zero_shot_spk_id=voice,
                    stream=False,
                    speed=speed,
                )
            else:
                gen = engine.inference_zero_shot(
                    text,
                    "",
                    None,
                    zero_shot_spk_id=voice,
                    stream=False,
                    speed=speed,
                )
        else:
            gen = engine.inference_sft(text, voice, stream=False, speed=speed)
        for audio_data in gen:
            all_segments.append(audio_data["tts_speech"].numpy())

    if not all_segments:
        raise RuntimeError("推理无输出")

    combined = (
        np.concatenate(all_segments, axis=1) if len(all_segments) > 1 else all_segments[0]
    )
    audio_1d = combined[0] if combined.ndim == 2 else combined
    wav_bytes = _audio_to_wav_bytes(audio_1d, native_sr)
    return wav_bytes, native_sr, sentences_info


# ---------------------------------------------------------------------------
# 音色管理
# ---------------------------------------------------------------------------


def voice_add(
    name: str, prompt_text: str, wav_path: Path
) -> Dict[str, Any]:
    if _cosyvoice_clone is None:
        raise RuntimeError(
            "clone 模型未加载, TTS_MODEL_MODE 必须 = all 或 clone"
        )

    import torchaudio

    info = torchaudio.info(str(wav_path))
    duration = info.num_frames / info.sample_rate
    if duration < 1.0:
        raise ValueError(f"音频过短 ({duration:.2f}s), 至少 1 秒")
    if duration > 30.0:
        logger.warning("音频较长 (%.2fs), 建议 ≤30s", duration)

    success = _cosyvoice_clone.add_zero_shot_spk(
        prompt_text, str(wav_path), name
    )
    if not success:
        raise RuntimeError(f"add_zero_shot_spk 返回 False: {name}")

    _cosyvoice_clone.save_spkinfo()

    record = {
        "name": name,
        "reference_text": prompt_text,
        "audio_file": wav_path.name,
        "file_size": os.path.getsize(wav_path),
        "audio_duration": duration,
        "added_at": datetime.datetime.now().isoformat(),
        "status": "active",
    }
    _registry["voices"][name] = record
    _save_registry()
    return record


def voice_remove(name: str) -> bool:
    if _cosyvoice_clone is None:
        return False
    spk2info = getattr(_cosyvoice_clone.frontend, "spk2info", {})
    if name in spk2info:
        del spk2info[name]
        _cosyvoice_clone.save_spkinfo()
    if name in _registry.get("voices", {}):
        del _registry["voices"][name]
        _save_registry()
        return True
    return False


def voice_list_all() -> List[str]:
    if _cosyvoice_clone is None:
        return list(PRESET_VOICES)
    return list(_cosyvoice_clone.frontend.spk2info.keys())


def voice_list_clone() -> List[str]:
    return list(_registry.get("voices", {}).keys())


def voice_get_info(name: str) -> Optional[Dict[str, Any]]:
    return _registry.get("voices", {}).get(name)


def voice_refresh_from_dir() -> Tuple[int, int]:
    """扫描 VOICES_DIR/*.txt + *.wav, 注册新音色. 返回 (success, total)"""
    if _cosyvoice_clone is None:
        return 0, 0

    pairs: List[Tuple[str, Path, Path]] = []
    for txt in VOICES_DIR.glob("*.txt"):
        wav = txt.with_suffix(".wav")
        if wav.exists():
            pairs.append((txt.stem, txt, wav))

    success = 0
    spk2info = getattr(_cosyvoice_clone.frontend, "spk2info", {})
    for name, txt_path, wav_path in pairs:
        if name in spk2info:
            continue
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            if not prompt_text:
                continue
            voice_add(name, prompt_text, wav_path)
            success += 1
        except Exception as exc:
            logger.warning("注册 %s 失败: %s", name, exc)
    return success, len(pairs)


def text_normalize(text: str) -> List[str]:
    engine = _cosyvoice_clone or _cosyvoice_sft
    if engine is None:
        return [text]
    return list(engine.frontend.text_normalize(text, split=True, text_frontend=True))


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_models()
        logger.info("CosyVoice 子服务就绪 (device=%s)", DEVICE)
    except Exception as exc:
        logger.error("CosyVoice 模型预加载失败: %s", exc, exc_info=True)
    yield


app = FastAPI(title="funspeech-cosyvoice-service", lifespan=lifespan)


def _check_token(request: Request) -> None:
    if not INTERNAL_SERVICE_TOKEN:
        return
    if request.headers.get("X-Internal-Token", "") != INTERNAL_SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="invalid internal token")


def _ws_check_token(websocket: WebSocket) -> bool:
    if not INTERNAL_SERVICE_TOKEN:
        return True
    return websocket.query_params.get("token", "") == INTERNAL_SERVICE_TOKEN


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "device": DEVICE,
        "mode": TTS_MODEL_MODE,
        "sft_loaded": _cosyvoice_sft is not None,
        "clone_loaded": _cosyvoice_clone is not None,
        "clone_model_version": _clone_model_actual_version
        if _cosyvoice_clone
        else None,
        "vllm": TTS_LOAD_VLLM,
        "trt": TTS_LOAD_TRT,
    }


@app.post("/tts/file")
async def tts_file(request: Request) -> Response:
    _check_token(request)
    body = await request.json()
    text = body.get("text") or ""
    voice = body.get("voice") or "中文女"
    speed = float(body.get("speed", 1.0))
    prompt = body.get("prompt") or ""
    return_timestamps = bool(body.get("return_timestamps", False))

    if not text:
        raise HTTPException(status_code=400, detail="text required")

    try:
        wav_bytes, native_sr, sentences = synthesize_offline(
            text=text,
            voice=voice,
            speed=speed,
            prompt=prompt,
            return_timestamps=return_timestamps,
        )
    except Exception as exc:
        logger.exception("合成失败")
        raise HTTPException(status_code=500, detail=f"synthesis: {exc}")

    headers = {"X-Native-Sample-Rate": str(native_sr)}
    if sentences is not None:
        headers["X-Sentences"] = json.dumps(sentences, ensure_ascii=False)
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.get("/voices")
async def voices_list(request: Request) -> dict:
    _check_token(request)
    all_voices = voice_list_all() if _cosyvoice_clone else list(PRESET_VOICES)
    return {
        "preset": [v for v in PRESET_VOICES if v in all_voices],
        "clone": voice_list_clone(),
        "all": all_voices,
        "registry": _registry.get("voices", {}),
    }


@app.get("/voices/{name}")
async def voice_info(request: Request, name: str) -> dict:
    _check_token(request)
    info = voice_get_info(name)
    if info is None:
        # 预设音色没有 registry 条目, 返回默认
        if name in PRESET_VOICES:
            return {"name": name, "type": "preset"}
        raise HTTPException(status_code=404, detail=f"voice not found: {name}")
    return {"name": name, "type": "clone", **info}


@app.post("/voices")
async def voice_create(
    request: Request,
    name: str = Form(...),
    prompt_text: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    _check_token(request)

    _ensure_voices_dirs()
    suffix = os.path.splitext(audio.filename or "voice.wav")[1] or ".wav"
    target = VOICES_DIR / f"{name}{suffix}"
    with open(target, "wb") as fp:
        fp.write(await audio.read())

    try:
        record = voice_add(name=name, prompt_text=prompt_text, wav_path=target)
    except Exception as exc:
        logger.exception("注册音色失败")
        raise HTTPException(status_code=400, detail=str(exc))
    return record


@app.delete("/voices/{name}")
async def voice_delete(request: Request, name: str) -> dict:
    _check_token(request)
    ok = voice_remove(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"voice not found: {name}")
    return {"removed": name}


@app.post("/voices/refresh")
async def voices_refresh(request: Request) -> dict:
    _check_token(request)
    success, total = voice_refresh_from_dir()
    return {"added": success, "total": total}


@app.post("/voices/reload")
async def voices_reload(request: Request) -> dict:
    """从磁盘热重载 spk2info + registry, 用于多副本同步.

    典型用法: cosyvoice-0 (主写副本) 写完音色后, 由网关广播到其它只读副本,
    其它副本调本接口立刻感知; 不重启进程.
    """
    _check_token(request)
    return reload_voices_from_disk()


@app.post("/text/normalize")
async def text_normalize_endpoint(request: Request) -> dict:
    _check_token(request)
    body = await request.json()
    text = body.get("text") or ""
    return {"sentences": text_normalize(text)}


# ---------------------------------------------------------------------------
# WebSocket 流式
#
# 协议:
#   client -> server (首帧 JSON, 一次合成一连接):
#     {text: str, voice: str, speed: float = 1.0, prompt: str = ""}
#     - 是否走 zero_shot / instruct2 / sft 由子服务根据 voice 是否在
#       clone 注册表中自动决定; prompt 非空且 voice 是 clone 时走 instruct2。
#   server -> client:
#     第 1 帧 JSON {type:"started", sample_rate}
#     之后 N 帧二进制 = float32 PCM mono 块(client 自己拼接 + 转格式)
#     最后 1 帧 JSON {type:"done"}
#   出错: {type:"error", message}
# ---------------------------------------------------------------------------


@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket) -> None:
    if not _ws_check_token(websocket):
        await websocket.close(code=4401, reason="invalid internal token")
        return

    await websocket.accept()

    try:
        first = await websocket.receive_text()
        params = json.loads(first)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"bad start: {exc}"})
        await websocket.close()
        return

    text = params.get("text") or ""
    voice = params.get("voice") or "中文女"
    speed = float(params.get("speed", 1.0))
    prompt = params.get("prompt") or ""

    if not text:
        await websocket.send_json({"type": "error", "message": "text required"})
        await websocket.close()
        return

    use_clone = _is_clone_voice(voice)
    if use_clone and _cosyvoice_clone is None:
        await websocket.send_json(
            {"type": "error", "message": "clone model not loaded"}
        )
        await websocket.close()
        return
    if not use_clone and _cosyvoice_sft is None:
        await websocket.send_json(
            {"type": "error", "message": "sft model not loaded"}
        )
        await websocket.close()
        return

    engine = _cosyvoice_clone if use_clone else _cosyvoice_sft
    native_sr = int(engine.sample_rate)
    await websocket.send_json({"type": "started", "sample_rate": native_sr})

    try:
        if use_clone:
            formatted_prompt = _format_prompt_for_clone(prompt)
            if prompt:
                gen = engine.inference_instruct2(
                    text,
                    formatted_prompt,
                    None,
                    zero_shot_spk_id=voice,
                    stream=True,
                    speed=speed,
                )
            else:
                gen = engine.inference_zero_shot(
                    text,
                    "",
                    None,
                    zero_shot_spk_id=voice,
                    stream=True,
                    speed=speed,
                )
        else:
            gen = engine.inference_sft(text, voice, stream=True, speed=speed)

        for audio_data in gen:
            chunk = audio_data["tts_speech"].numpy()
            if chunk.ndim == 2:
                chunk = chunk[0]
            chunk_f32 = np.asarray(chunk, dtype=np.float32)
            await websocket.send_bytes(chunk_f32.tobytes())

        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("TTS WS 客户端断开")
    except Exception as exc:
        logger.exception("流式合成失败")
        try:
            await websocket.send_json(
                {"type": "error", "message": f"synthesis: {exc}"}
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8004")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
