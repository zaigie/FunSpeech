# -*- coding: utf-8 -*-
"""FunSpeech-compatible facade for vLLM-Omni TTS servers.

The facade keeps the existing subservice protocol stable:
  - GET  /health
  - POST /tts/file
  - WS   /tts/stream
  - GET/POST/DELETE /voices

Internally it talks to vLLM-Omni's OpenAI-compatible Speech API:
  - POST /v1/audio/speech
  - GET/POST/DELETE /v1/audio/voices
"""

from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import io
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
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
from fastapi.responses import JSONResponse, Response


logger = logging.getLogger("vllm_omni_tts_facade")

_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9一-鿿._-]{1,64}$")


@dataclass
class OmniTTSConfig:
    service_name: str
    public_title: str
    model_id: str
    task_type: str = ""
    default_voice: str = "vivian"
    default_language: str = "Auto"
    sample_rate: int = 24000
    supports_presets: bool = False
    supports_voice_clone: bool = True
    include_task_type: bool = True
    prefer_uploaded_voice: bool = True
    require_ref_for_voice: bool = False
    deploy_config_name: str = ""
    voices_dir: Path = Path("/app/voices")
    port: int = 8005
    internal_token: str = ""
    api_base: str = "http://127.0.0.1:8091"
    api_key: str = "EMPTY"
    start_omni: bool = True
    omni_host: str = "127.0.0.1"
    omni_port: int = 8091
    startup_timeout_sec: float = 900.0
    request_timeout_sec: float = 300.0
    stream_chunk_sec: float = 0.2
    serve_command: str = "vllm"
    gpu_memory_utilization: str = ""
    stage_overrides: str = ""
    extra_serve_args: str = ""
    trust_remote_code: bool = True
    enforce_eager: bool = False
    no_async_chunk: bool = False
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def registry_file(self) -> Path:
        return self.voices_dir / "voice_registry.json"


class OmniTTSFacade:
    def __init__(self, config: OmniTTSConfig):
        self.config = config
        self._registry: Dict[str, Any] = {"version": "vllm-omni-tts-v1", "voices": {}}
        self._load_failed = False
        self._load_error_msg = ""
        self._omni_proc: Optional[subprocess.Popen[str]] = None
        self._ready = False
        self.app = self._create_app()

    # ------------------------------------------------------------------ setup

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                self._ensure_dirs()
                self._registry = self._load_registry_from_disk()
                if self.config.start_omni:
                    await asyncio.to_thread(self._start_omni_server)
                else:
                    self._ready = self._check_omni_ready(timeout=1.0)
            except Exception as exc:
                self._load_failed = True
                self._load_error_msg = str(exc)
                logger.error("%s 启动失败: %s", self.config.service_name, exc, exc_info=True)
            yield
            await asyncio.to_thread(self._stop_omni_server)

        app = FastAPI(title=self.config.public_title, lifespan=lifespan)
        self._mount_routes(app)
        return app

    def _ensure_dirs(self) -> None:
        self.config.voices_dir.mkdir(parents=True, exist_ok=True)

    def _start_omni_server(self) -> None:
        if self._omni_proc is not None and self._omni_proc.poll() is None:
            return

        cmd = self._build_omni_command()
        env = os.environ.copy()
        env.update(self.config.env)
        env.setdefault("SPEAKER_SAMPLES_DIR", str(self.config.voices_dir / "speakers"))
        Path(env["SPEAKER_SAMPLES_DIR"]).mkdir(parents=True, exist_ok=True)

        logger.info("启动 vLLM-Omni: %s", " ".join(shlex.quote(x) for x in cmd))
        self._omni_proc = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
            text=True,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        self._ready = self._wait_for_omni_ready()

    def _stop_omni_server(self) -> None:
        proc = self._omni_proc
        if proc is None or proc.poll() is not None:
            return
        logger.info("停止 vLLM-Omni pid=%s", proc.pid)
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            logger.warning("vLLM-Omni 未正常退出, kill pid=%s", proc.pid)
            try:
                proc.kill()
            except Exception:
                pass

    def _build_omni_command(self) -> List[str]:
        cmd = [
            self.config.serve_command,
            "serve",
            self.config.model_id,
            "--host",
            self.config.omni_host,
            "--port",
            str(self.config.omni_port),
            "--omni",
        ]
        deploy_config = self._resolve_deploy_config()
        if deploy_config:
            cmd.extend(["--deploy-config", deploy_config])
        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.config.enforce_eager:
            cmd.append("--enforce-eager")
        if self.config.no_async_chunk:
            cmd.append("--no-async-chunk")
        if self.config.gpu_memory_utilization:
            cmd.extend(["--gpu-memory-utilization", self.config.gpu_memory_utilization])
        if self.config.stage_overrides:
            cmd.extend(["--stage-overrides", self.config.stage_overrides])
        if self.config.extra_serve_args:
            cmd.extend(shlex.split(self.config.extra_serve_args))
        return cmd

    def _resolve_deploy_config(self) -> str:
        name = self.config.deploy_config_name
        if not name:
            return ""
        path = Path(name)
        if path.exists():
            return str(path)
        try:
            import importlib.resources as resources

            candidate = resources.files("vllm_omni").joinpath("deploy", name)
            if candidate.is_file():
                return str(candidate)
        except Exception:
            pass
        return name

    def _wait_for_omni_ready(self) -> bool:
        deadline = time.time() + self.config.startup_timeout_sec
        last_error = ""
        while time.time() < deadline:
            if self._omni_proc is not None and self._omni_proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM-Omni exited early with code {self._omni_proc.returncode}"
                )
            ok, err = self._probe_omni()
            if ok:
                logger.info("vLLM-Omni ready: %s", self.config.api_base)
                return True
            last_error = err
            time.sleep(2.0)
        raise RuntimeError(f"vLLM-Omni not ready: {last_error}")

    def _check_omni_ready(self, timeout: float = 5.0) -> bool:
        ok, _ = self._probe_omni(timeout=timeout)
        return ok

    def _probe_omni(self, timeout: float = 5.0) -> Tuple[bool, str]:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{self.config.api_base}/v1/audio/voices")
            return resp.status_code == 200, f"status={resp.status_code} {resp.text[:200]}"
        except Exception as exc:
            return False, str(exc)

    # ---------------------------------------------------------------- registry

    def _load_registry_from_disk(self) -> Dict[str, Any]:
        path = self.config.registry_file
        if not path.exists():
            return {"version": "vllm-omni-tts-v1", "voices": {}}
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if not isinstance(data, dict):
                raise ValueError("registry root is not object")
            data.setdefault("version", "vllm-omni-tts-v1")
            data.setdefault("voices", {})
            return data
        except Exception as exc:
            logger.warning("加载 voice registry 失败: %s", exc)
            return {"version": "vllm-omni-tts-v1", "voices": {}}

    def _save_registry(self) -> None:
        self._ensure_dirs()
        now = _dt.datetime.now().isoformat()
        self._registry["updated_at"] = now
        self._registry.setdefault("created_at", now)
        tmp = self.config.registry_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(self._registry, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, self.config.registry_file)

    def _voice_audio_path(self, name: str, record: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        record = record or self._registry.get("voices", {}).get(name) or {}
        audio_file = record.get("audio_file")
        if not audio_file:
            return None
        path = (self.config.voices_dir / Path(audio_file).name).resolve()
        try:
            path.relative_to(self.config.voices_dir.resolve())
        except (ValueError, OSError):
            return None
        return path if path.exists() else None

    # ------------------------------------------------------------------ routes

    def _mount_routes(self, app: FastAPI) -> None:
        @app.get("/health", response_model=None)
        async def health() -> Any:
            body = self._health_body()
            if body["status"] == "healthy":
                return body
            return JSONResponse(status_code=503, content=body)

        @app.post("/tts/file")
        async def tts_file(request: Request) -> Response:
            self._check_token(request)
            body = await request.json()
            text = body.get("text") or ""
            if not text:
                raise HTTPException(status_code=400, detail="text required")
            voice = body.get("voice") or self.config.default_voice
            speed = float(body.get("speed", 1.0))
            prompt = body.get("prompt") or ""
            return_timestamps = bool(body.get("return_timestamps", False))
            language = body.get("language")

            try:
                wav_bytes, native_sr, sentences = await asyncio.to_thread(
                    self.synthesize_offline,
                    text,
                    voice,
                    speed,
                    prompt,
                    return_timestamps,
                    language,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("vLLM-Omni 合成失败")
                raise HTTPException(status_code=500, detail=f"synthesis: {exc}") from exc

            headers = {"X-Native-Sample-Rate": str(native_sr)}
            if sentences is not None:
                headers["X-Sentences"] = json.dumps(sentences, ensure_ascii=True)
            return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

        @app.get("/voices")
        async def voices(request: Request) -> Dict[str, Any]:
            self._check_token(request)
            return self.voices_listing()

        @app.get("/voices/{name}")
        async def voice_info(request: Request, name: str) -> Dict[str, Any]:
            self._check_token(request)
            self._validate_voice_name(name)
            listing = self.voices_listing()
            if name not in listing.get("all", []):
                raise HTTPException(status_code=404, detail=f"voice not found: {name}")
            info = listing.get("info", {}).get(name, {"name": name})
            record = listing.get("registry", {}).get(name, {})
            return {**info, **record, "name": name}

        @app.post("/voices")
        async def voice_create(
            request: Request,
            name: str = Form(...),
            prompt_text: str = Form(""),
            audio: UploadFile = File(...),
        ) -> Dict[str, Any]:
            self._check_token(request)
            self._validate_voice_name(name)
            self._ensure_ready_or_503()
            self._ensure_dirs()

            suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(audio.filename or "voice.wav").suffix)[:8] or ".wav"
            target = self.config.voices_dir / f"{name}{suffix}"
            content = await audio.read()
            with open(target, "wb") as fp:
                fp.write(content)
            txt_path = self.config.voices_dir / f"{name}.txt"
            txt_path.write_text(prompt_text.strip(), encoding="utf-8")

            try:
                return await asyncio.to_thread(self.register_voice, name, prompt_text, target)
            except Exception as exc:
                for path in (target, txt_path):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @app.delete("/voices/{name}")
        async def voice_delete(request: Request, name: str) -> Dict[str, Any]:
            self._check_token(request)
            self._validate_voice_name(name)
            removed = await asyncio.to_thread(self.delete_voice, name)
            if not removed:
                raise HTTPException(status_code=404, detail=f"voice not found: {name}")
            return {"removed": name}

        @app.post("/voices/refresh")
        async def voices_refresh(request: Request) -> Dict[str, Any]:
            self._check_token(request)
            try:
                added, total = await asyncio.to_thread(self.refresh_voices_from_dir)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"added": added, "total": total}

        @app.post("/voices/reload")
        async def voices_reload(request: Request) -> Dict[str, Any]:
            self._check_token(request)
            self._registry = self._load_registry_from_disk()
            added, total = await asyncio.to_thread(self.refresh_voices_from_dir)
            return {
                "clone_voices": len(self._registry.get("voices", {})),
                "registry_voices": len(self._registry.get("voices", {})),
                "reuploaded": added,
                "total": total,
                "clone_loaded": self.config.supports_voice_clone,
            }

        @app.post("/text/normalize")
        async def text_normalize(request: Request) -> Dict[str, Any]:
            self._check_token(request)
            body = await request.json()
            text = body.get("text") or ""
            parts = [p for p in re.split(r"(?<=[。！？.!?])\s*", text) if p]
            return {"sentences": parts or [text]}

        @app.websocket("/tts/stream")
        async def tts_stream(websocket: WebSocket) -> None:
            if not self._ws_check_token(websocket):
                await websocket.close(code=4401, reason="invalid internal token")
                return
            await websocket.accept()
            try:
                first = await websocket.receive_text()
                params = json.loads(first)
                text = params.get("text") or ""
                if not text:
                    await websocket.send_json({"type": "error", "message": "text required"})
                    await websocket.close()
                    return
                voice = params.get("voice") or self.config.default_voice
                speed = float(params.get("speed", 1.0))
                prompt = params.get("prompt") or ""
                language = params.get("language")

                await websocket.send_json({"type": "started", "sample_rate": self.config.sample_rate})
                async for chunk in self.stream_pcm_float32(text, voice, speed, prompt, language):
                    if websocket.client_state.name != "CONNECTED":
                        break
                    await websocket.send_bytes(chunk.tobytes())
                await websocket.send_json({"type": "done"})
            except WebSocketDisconnect:
                logger.info("vLLM-Omni WS 客户端断开")
            except Exception as exc:
                logger.exception("vLLM-Omni 流式合成失败")
                try:
                    await websocket.send_json({"type": "error", "message": f"synthesis: {exc}"})
                except Exception:
                    pass
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

    # ---------------------------------------------------------------- behavior

    def _health_body(self) -> Dict[str, Any]:
        ready = self._check_omni_ready(timeout=1.5)
        self._ready = ready
        status = "healthy" if ready else "starting"
        if self._load_failed:
            status = "unhealthy"
        body: Dict[str, Any] = {
            "status": status,
            "service": self.config.service_name,
            "backend": "vllm-omni",
            "model_id": self.config.model_id,
            "task_type": self.config.task_type,
            "api_base": self.config.api_base,
            "model_loaded": ready,
            "sft_loaded": self.config.supports_presets or self.config.task_type.lower() in ("customvoice", "voicedesign"),
            "clone_loaded": self.config.supports_voice_clone,
            "sample_rate": self.config.sample_rate,
            "voices_dir": str(self.config.voices_dir),
            "clone_voices": len(self._registry.get("voices", {})),
        }
        if self._omni_proc is not None:
            body["omni_pid"] = self._omni_proc.pid
            body["omni_returncode"] = self._omni_proc.poll()
        if self._load_failed:
            body["error"] = self._load_error_msg
        return body

    def synthesize_offline(
        self,
        text: str,
        voice: str,
        speed: float,
        prompt: str,
        return_timestamps: bool,
        language: Optional[str] = None,
    ) -> Tuple[bytes, int, Optional[List[Dict[str, str]]]]:
        self._ensure_ready_or_503()
        payload = self._speech_payload(
            text=text,
            voice=voice,
            speed=speed,
            prompt=prompt,
            language=language,
            response_format="wav",
            stream=False,
        )
        with httpx.Client(timeout=self.config.request_timeout_sec) as client:
            resp = client.post(
                f"{self.config.api_base}/v1/audio/speech",
                json=payload,
                headers=self._omni_headers(),
            )
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM-Omni speech API {resp.status_code}: {resp.text[:500]}")

        wav_bytes = resp.content
        native_sr, samples = self._inspect_wav(wav_bytes)
        sentences = self._estimate_sentences(text, samples, native_sr) if return_timestamps else None
        return wav_bytes, native_sr, sentences

    async def stream_pcm_float32(
        self,
        text: str,
        voice: str,
        speed: float,
        prompt: str,
        language: Optional[str] = None,
    ):
        self._ensure_ready_or_503()
        payload = self._speech_payload(
            text=text,
            voice=voice,
            speed=speed,
            prompt=prompt,
            language=language,
            response_format="pcm",
            stream=True,
        )
        timeout = httpx.Timeout(self.config.request_timeout_sec, read=None)
        leftover = b""
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.config.api_base}/v1/audio/speech",
                json=payload,
                headers=self._omni_headers(),
            ) as resp:
                if resp.status_code != 200:
                    text_body = await resp.aread()
                    raise RuntimeError(
                        f"vLLM-Omni stream API {resp.status_code}: {text_body[:500]!r}"
                    )
                async for part in resp.aiter_bytes():
                    if not part:
                        continue
                    data = leftover + part
                    if len(data) % 2:
                        leftover = data[-1:]
                        data = data[:-1]
                    else:
                        leftover = b""
                    if not data:
                        continue
                    pcm_i16 = np.frombuffer(data, dtype=np.int16)
                    if pcm_i16.size:
                        yield (pcm_i16.astype(np.float32) / 32768.0)

    def _speech_payload(
        self,
        text: str,
        voice: str,
        speed: float,
        prompt: str,
        language: Optional[str],
        response_format: str,
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model_id,
            "input": text,
            "response_format": response_format,
            "speed": speed,
        }
        if stream:
            payload["stream"] = True
        if self.config.include_task_type and self.config.task_type:
            payload["task_type"] = self.config.task_type
        lang = (language or self.config.default_language or "").strip()
        if lang and lang.lower() not in ("none", "null"):
            payload["language"] = lang
        if prompt:
            payload["instructions"] = prompt

        record = self._registry.get("voices", {}).get(voice)
        if record and self.config.require_ref_for_voice:
            ref_path = self._voice_audio_path(voice, record)
            if ref_path is None:
                raise RuntimeError(f"voice audio not found: {voice}")
            payload["ref_audio"] = self._audio_to_data_url(ref_path)
            payload["ref_text"] = record.get("reference_text", "")
            payload["voice"] = voice
        elif record and self.config.prefer_uploaded_voice:
            payload["voice"] = voice
        elif self.config.supports_presets or self.config.task_type.lower() in ("customvoice", "voicedesign"):
            payload["voice"] = voice
        else:
            raise RuntimeError(f"voice not found: {voice}")
        return payload

    def voices_listing(self) -> Dict[str, Any]:
        self._registry = self._load_registry_from_disk()
        upstream = self._get_upstream_voices()
        upstream_voices = list(upstream.get("voices") or [])
        uploaded_entries = upstream.get("uploaded_voices") or []
        uploaded_names = {
            item.get("name")
            for item in uploaded_entries
            if isinstance(item, dict) and item.get("name")
        }
        registry_names = {
            name
            for name, record in (self._registry.get("voices") or {}).items()
            if self._voice_audio_path(name, record) is not None
        }
        clone = sorted((uploaded_names | registry_names))
        preset = [v for v in upstream_voices if v not in set(clone)]
        if not self.config.supports_presets and self.config.task_type.lower() not in ("customvoice", "voicedesign"):
            preset = []
        all_voices = list(dict.fromkeys(preset + clone))

        registry = dict(self._registry.get("voices") or {})
        for item in uploaded_entries:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = item["name"]
            registry.setdefault(
                name,
                {
                    "name": name,
                    "reference_text": item.get("ref_text", ""),
                    "audio_file": "",
                    "added_at": item.get("created_at", ""),
                    "status": "active",
                },
            )

        info: Dict[str, Dict[str, Any]] = {}
        for voice in all_voices:
            if voice in clone:
                record = registry.get(voice, {})
                info[voice] = {
                    "name": voice,
                    "type": "clone",
                    "language": "multilingual",
                    "gender": "unknown",
                    "description": f"{self.config.service_name} clone voice: {voice}",
                    "sample_rate": self.config.sample_rate,
                    "available": True,
                    "reference_text": record.get("reference_text", record.get("ref_text", "")),
                    "audio_file": record.get("audio_file", ""),
                    "added_at": record.get("added_at", ""),
                }
            else:
                info[voice] = {
                    "name": voice,
                    "type": "preset",
                    "language": "multilingual",
                    "gender": "unknown",
                    "description": f"{self.config.service_name} preset voice: {voice}",
                    "sample_rate": self.config.sample_rate,
                    "available": True,
                }
        return {
            "preset": preset,
            "clone": clone,
            "all": all_voices,
            "registry": registry,
            "info": info,
            "sample_rate": self.config.sample_rate,
            "backend": "vllm-omni",
            "model_id": self.config.model_id,
            "task_type": self.config.task_type,
            "upstream": upstream,
        }

    def register_voice(self, name: str, prompt_text: str, wav_path: Path) -> Dict[str, Any]:
        self._validate_voice_name(name)
        self._ensure_ready_or_503()
        if not self.config.supports_voice_clone:
            raise RuntimeError(f"{self.config.service_name} does not support voice upload")
        duration = self._audio_duration(wav_path)
        if duration < 1.0:
            raise ValueError(f"音频过短 ({duration:.2f}s), 至少 1 秒")

        with open(wav_path, "rb") as fp:
            files = {"audio_sample": (wav_path.name, fp, self._mime_type(wav_path))}
            data = {
                "name": name,
                "consent": f"funspeech-{name}",
                "ref_text": prompt_text.strip(),
                "speaker_description": f"FunSpeech clone voice {name}",
            }
            with httpx.Client(timeout=self.config.request_timeout_sec) as client:
                resp = client.post(
                    f"{self.config.api_base}/v1/audio/voices",
                    files=files,
                    data=data,
                    headers=self._omni_headers(include_content_type=False),
                )
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM-Omni voice upload {resp.status_code}: {resp.text[:500]}")

        record = {
            "name": name,
            "reference_text": prompt_text.strip(),
            "audio_file": wav_path.name,
            "file_size": os.path.getsize(wav_path),
            "audio_duration": duration,
            "added_at": _dt.datetime.now().isoformat(),
            "status": "active",
            "backend": "vllm-omni",
        }
        self._registry.setdefault("voices", {})[name] = record
        self._save_registry()
        return record

    def delete_voice(self, name: str) -> bool:
        removed = False
        try:
            with httpx.Client(timeout=self.config.request_timeout_sec) as client:
                resp = client.delete(
                    f"{self.config.api_base}/v1/audio/voices/{name}",
                    headers=self._omni_headers(),
                )
            removed = resp.status_code == 200
        except Exception as exc:
            logger.warning("删除 vLLM-Omni voice 失败: %s", exc)

        record = self._registry.get("voices", {}).pop(name, None)
        if record:
            removed = True
            self._save_registry()
            for path in (
                self.config.voices_dir / f"{name}.txt",
                self._voice_audio_path(name, record),
            ):
                if path and path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass
        return removed

    def refresh_voices_from_dir(self) -> Tuple[int, int]:
        self._ensure_ready_or_503()
        self._ensure_dirs()
        pairs: List[Tuple[str, Path, Path]] = []
        for txt in self.config.voices_dir.glob("*.txt"):
            wav = txt.with_suffix(".wav")
            if wav.exists():
                pairs.append((txt.stem, txt, wav))

        added = 0
        existing = set((self._registry.get("voices") or {}).keys())
        for name, txt_path, wav_path in pairs:
            if name in existing:
                continue
            prompt_text = txt_path.read_text(encoding="utf-8").strip()
            try:
                self.register_voice(name, prompt_text, wav_path)
                added += 1
            except Exception as exc:
                logger.warning("刷新音色 %s 失败: %s", name, exc)
        return added, len(pairs)

    # ---------------------------------------------------------------- helpers

    def _check_token(self, request: Request) -> None:
        token = self.config.internal_token
        if not token:
            return
        if request.headers.get("X-Internal-Token", "") != token:
            raise HTTPException(status_code=401, detail="invalid internal token")

    def _ws_check_token(self, websocket: WebSocket) -> bool:
        token = self.config.internal_token
        if not token:
            return True
        return websocket.query_params.get("token", "") == token

    def _ensure_ready_or_503(self) -> None:
        if self._load_failed:
            raise HTTPException(status_code=503, detail=self._load_error_msg)
        if not self._ready and not self._check_omni_ready(timeout=2.0):
            raise HTTPException(status_code=503, detail="vLLM-Omni server is not ready")
        self._ready = True

    def _validate_voice_name(self, name: str) -> None:
        if not name or not _VOICE_NAME_RE.fullmatch(name):
            raise HTTPException(
                status_code=400,
                detail=(
                    "invalid voice name: only A-Za-z0-9, 中文, '.', '_', '-' allowed, "
                    "length 1-64"
                ),
            )

    def _omni_headers(self, include_content_type: bool = True) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _get_upstream_voices(self) -> Dict[str, Any]:
        if not self._ready and not self._check_omni_ready(timeout=1.0):
            return {"voices": [], "uploaded_voices": []}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self.config.api_base}/v1/audio/voices",
                    headers=self._omni_headers(include_content_type=False),
                )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    data.setdefault("voices", [])
                    data.setdefault("uploaded_voices", [])
                    return data
        except Exception as exc:
            logger.debug("拉取 vLLM-Omni voices 失败: %s", exc)
        return {"voices": [], "uploaded_voices": []}

    def _audio_to_data_url(self, path: Path) -> str:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{self._mime_type(path)};base64,{payload}"

    def _audio_duration(self, path: Path) -> float:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)

    def _inspect_wav(self, wav_bytes: bytes) -> Tuple[int, int]:
        try:
            info = sf.info(io.BytesIO(wav_bytes))
            return int(info.samplerate), int(info.frames)
        except Exception:
            return self.config.sample_rate, 0

    def _estimate_sentences(self, text: str, sample_count: int, sample_rate: int) -> List[Dict[str, str]]:
        parts = [p for p in re.split(r"(?<=[。！？.!?])\s*", text) if p]
        if not parts:
            return []
        total_chars = max(sum(len(p) for p in parts), 1)
        total_ms = sample_count / max(sample_rate, 1) * 1000
        cursor = 0.0
        out: List[Dict[str, str]] = []
        for part in parts:
            dur = total_ms * (len(part) / total_chars)
            out.append(
                {
                    "text": part,
                    "begin_time": str(int(cursor)),
                    "end_time": str(int(cursor + dur)),
                }
            )
            cursor += dur
        return out

    def _mime_type(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".mp3":
            return "audio/mpeg"
        if ext == ".flac":
            return "audio/flac"
        if ext == ".ogg":
            return "audio/ogg"
        if ext in (".m4a", ".mp4"):
            return "audio/mp4"
        return "audio/wav"


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def first_env(names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default
