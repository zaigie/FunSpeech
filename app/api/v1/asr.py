# -*- coding: utf-8 -*-
"""
ASR API路由
"""

from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends,
    Body,
    Query,
    Header,
    File,
    UploadFile,
)
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, Annotated
import logging

from ...core.config import settings
from ...core.exceptions import (
    ASRException,
    AuthenticationException,
    InvalidParameterException,
    InvalidMessageException,
    UnsupportedSampleRateException,
)
from ...core.security import validate_xls_token, mask_sensitive_data
from ...models.asr import (
    ASRResponse,
    ASRHealthCheckResponse,
    ASRModelsResponse,
    ASRSuccessResponse,
    ASRErrorResponse,
    ASRQueryParams,
    ASRHeaders,
)
from ...utils.common import (
    generate_task_id,
    validate_appkey,
)
from ...utils.audio import (
    validate_audio_format,
    validate_sample_rate,
    download_audio_from_url,
    save_audio_to_temp_file,
    cleanup_temp_file,
    get_audio_file_suffix,
    normalize_audio_for_asr,
)
from ...services.asr.manager import get_model_manager

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/stream/v1", tags=["ASR"])


async def get_asr_params(request: Request) -> ASRQueryParams:
    """从请求中提取并验证ASR参数"""
    # 从URL查询参数中获取
    query_params = dict(request.query_params)

    # 创建ASRQueryParams实例，Pydantic会自动验证和设置默认值
    try:
        return ASRQueryParams(**query_params)
    except Exception as e:
        raise InvalidParameterException(f"请求参数错误: {str(e)}")


@router.post(
    "/asr",
    response_model=ASRResponse,
    responses={
        200: {
            "description": "识别成功",
            "model": ASRSuccessResponse,
        },
        400: {
            "description": "请求参数错误",
            "model": ASRErrorResponse,
        },
        500: {"description": "服务器内部错误", "model": ASRErrorResponse},
    },
    summary="一句话识别",
    description="语音识别RESTful API",
    openapi_extra={
        "parameters": [
            {
                "name": "appkey",
                "in": "query",
                "required": True,
                "schema": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "example": "your_app_key_here",
                },
                "description": "应用Appkey，用于API调用认证",
            },
            {
                "name": "format",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "enum": [
                        "pcm",
                        "wav",
                        "opus",
                        "speex",
                        "amr",
                        "mp3",
                        "aac",
                        "m4a",
                        "flac",
                        "ogg",
                    ],
                    "default": "pcm",
                    "example": "pcm",
                },
                "description": "音频格式。支持: pcm, wav, opus, speex, amr, mp3, aac, m4a, flac, ogg。仅在使用audio_address参数时生效，使用二进制音频流时默认为wav格式",
            },
            {
                "name": "sample_rate",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "integer",
                    "enum": [8000, 16000, 22050, 44100, 48000],
                    "default": 16000,
                    "example": 16000,
                },
                "description": "音频采样率（Hz）。支持: 8000, 16000, 22050, 44100, 48000",
            },
            {
                "name": "vocabulary_id",
                "in": "query",
                "required": False,
                "schema": {"type": "string", "maxLength": 32, "example": "vocab_12345"},
                "description": "热词表ID，用于提高特定词汇的识别准确率",
            },
            {
                "name": "customization_id",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 64,
                    "default": "paraformer-large",
                    "example": "paraformer-large",
                },
                "description": "自定义模型ID，指定使用的ASR模型",
            },
            {
                "name": "enable_punctuation_prediction",
                "in": "query",
                "required": False,
                "schema": {"type": "boolean", "default": False, "example": True},
                "description": "是否启用标点符号预测",
            },
            {
                "name": "enable_inverse_text_normalization",
                "in": "query",
                "required": False,
                "schema": {"type": "boolean", "default": False, "example": False},
                "description": "是否启用反向文本标准化（将中文数字转为阿拉伯数字）",
            },
            {
                "name": "enable_voice_detection",
                "in": "query",
                "required": False,
                "schema": {"type": "boolean", "default": False, "example": False},
                "description": "是否启用语音活动检测（VAD）",
            },
            {
                "name": "disfluency",
                "in": "query",
                "required": False,
                "schema": {"type": "boolean", "default": False, "example": False},
                "description": "是否过滤语气词（嗯、啊等）",
            },
            {
                "name": "audio_address",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 512,
                    "example": "https://example.com/audio.wav",
                },
                "description": "音频文件下载链接（HTTP/HTTPS）",
            },
            {
                "name": "dolphin_lang_sym",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 8,
                    "default": "zh",
                    "example": "zh",
                },
                "description": "Dolphin引擎语言符号",
            },
            {
                "name": "dolphin_region_sym",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 16,
                    "default": "SHANGHAI",
                    "example": "SHANGHAI",
                },
                "description": "Dolphin引擎区域符号",
            },
            {
                "name": "X-NLS-Token",
                "in": "header",
                "required": False,
                "schema": {
                    "type": "string",
                    "minLength": 10,
                    "maxLength": 256,
                    "example": "your_access_token_here",
                },
                "description": "访问令牌，用于身份认证",
            },
        ],
        "requestBody": {
            "description": "音频文件的二进制数据（当不使用audio_address参数时）",
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "required": False,
        },
    },
)
async def asr_transcribe(
    request: Request, params: Annotated[ASRQueryParams, Depends(get_asr_params)]
) -> JSONResponse:
    """语音识别API端点"""
    task_id = generate_task_id()
    audio_path = None

    try:
        # 验证请求头部（鉴权）
        token = validate_xls_token(request, task_id)
        logger.info(
            f"[{task_id}] 请求验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

        logger.info(f"[{task_id}] 请求参数: {params}")

        # 验证format参数（如果指定了的话）
        if params.format and not validate_audio_format(params.format):
            raise InvalidParameterException(
                f"不支持的音频格式: {params.format}。支持的格式: {', '.join(settings.SUPPORTED_AUDIO_FORMATS)}",
                task_id,
            )

        # 验证sample_rate参数
        if params.sample_rate and not validate_sample_rate(params.sample_rate):
            raise InvalidParameterException(
                f"不支持的采样率: {params.sample_rate}。支持的采样率: {', '.join(map(str, settings.SUPPORTED_SAMPLE_RATES))}",
                task_id,
            )

        # 获取音频数据
        if params.audio_address:
            # 方式1: 从URL下载音频
            logger.info(f"[{task_id}] 从URL下载音频: {params.audio_address}")
            audio_data = download_audio_from_url(params.audio_address)

            # 使用format参数指定的格式保存文件
            file_suffix = get_audio_file_suffix(params.audio_address, params.format)
            logger.info(
                f"[{task_id}] 使用audio_address，format参数生效: {params.format}，文件后缀: {file_suffix}"
            )
            audio_path = save_audio_to_temp_file(audio_data, file_suffix)

        else:
            # 方式2: 从请求体读取二进制音频数据
            logger.info(f"[{task_id}] 从请求体读取音频数据")

            # 读取请求体
            audio_data = await request.body()
            if not audio_data:
                raise InvalidMessageException("音频数据为空", task_id)

            # 检查文件大小
            if len(audio_data) > settings.MAX_AUDIO_SIZE:
                max_size_mb = settings.MAX_AUDIO_SIZE // 1024 // 1024
                raise InvalidMessageException(
                    f"音频文件太大，最大支持{max_size_mb}MB", task_id
                )

            # 使用二进制音频流时，format参数不生效，默认为wav格式
            file_suffix = get_audio_file_suffix(None, None)
            logger.info(
                f"[{task_id}] 使用二进制音频流，format参数不生效，默认使用wav格式，文件后缀: {file_suffix}"
            )
            audio_path = save_audio_to_temp_file(audio_data, file_suffix)

        logger.info(f"[{task_id}] 音频文件已保存: {audio_path}")

        # 将音频标准化为ASR模型所需的格式（统一转换为WAV格式，指定采样率）
        normalized_audio_path = normalize_audio_for_asr(audio_path, params.sample_rate)
        logger.info(f"[{task_id}] 音频已标准化: {normalized_audio_path}")

        # 执行语音识别
        model_manager = get_model_manager()
        asr_engine = model_manager.get_asr_engine(params.customization_id)

        # 准备热词（如果有vocabulary_id，这里可以根据ID查询热词）
        hotwords = ""  # 实际项目中可以根据vocabulary_id查询对应的热词

        result_text = asr_engine.transcribe_file(
            audio_path=normalized_audio_path,
            hotwords=hotwords,
            enable_punctuation=params.enable_punctuation_prediction,
            enable_itn=params.enable_inverse_text_normalization,
            enable_vad=params.enable_voice_detection,
            sample_rate=params.sample_rate,
            dolphin_lang_sym=params.dolphin_lang_sym,
            dolphin_region_sym=params.dolphin_region_sym,
        )

        logger.info(f"[{task_id}] 识别完成: {result_text}")

        # 返回成功响应
        response_data = {
            "task_id": task_id,
            "result": result_text,
            "status": 20000000,
            "message": "SUCCESS",
        }

        return JSONResponse(content=response_data, headers={"task_id": task_id})

    except ASRException as e:
        e.task_id = task_id
        logger.error(f"[{task_id}] ASR异常: {e.message}")
        response_data = {
            "task_id": task_id,
            "result": "",
            "status": e.status_code,
            "message": e.message,
        }
        return JSONResponse(content=response_data, headers={"task_id": task_id})

    except Exception as e:
        logger.error(f"[{task_id}] 未知异常: {str(e)}")
        response_data = {
            "task_id": task_id,
            "result": "",
            "status": 50000000,
            "message": f"内部服务错误: {str(e)}",
        }
        return JSONResponse(content=response_data, headers={"task_id": task_id})

    finally:
        # 清理临时文件
        if audio_path:
            cleanup_temp_file(audio_path)
        if (
            "normalized_audio_path" in locals()
            and normalized_audio_path
            and normalized_audio_path != audio_path
        ):
            cleanup_temp_file(normalized_audio_path)


@router.get(
    "/asr/health",
    response_model=ASRHealthCheckResponse,
    summary="ASR服务健康检查",
    description="检查语音识别服务的运行状态",
)
async def health_check(request: Request):
    """ASR服务健康检查端点"""
    try:
        # 验证请求头部（鉴权）
        token = validate_xls_token(request)
        logger.info(
            f"健康检查请求验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

        model_manager = get_model_manager()

        # 尝试获取默认模型的引擎
        try:
            asr_engine = model_manager.get_asr_engine()
            model_loaded = asr_engine.is_model_loaded()
            device = asr_engine.device
        except Exception:
            model_loaded = False
            device = "unknown"

        memory_info = model_manager.get_memory_usage()

        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "device": device,
            "version": settings.APP_VERSION,
            "message": (
                "ASR service is running normally"
                if model_loaded
                else "ASR model not loaded"
            ),
            "loaded_models": memory_info["model_list"],
            "memory_usage": memory_info.get("gpu_memory"),
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "device": "unknown",
            "version": settings.APP_VERSION,
            "message": str(e),
        }


@router.get(
    "/asr/models",
    response_model=ASRModelsResponse,
    summary="获取可用模型列表",
    description="返回系统中所有可用的ASR模型信息",
)
async def list_models(request: Request):
    """获取可用模型列表端点"""
    try:
        # 验证请求头部（鉴权）
        token = validate_xls_token(request)
        logger.info(
            f"模型列表请求验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

        model_manager = get_model_manager()
        models = model_manager.list_models()

        loaded_count = sum(1 for model in models if model["loaded"])

        return {
            "models": models,
            "total": len(models),
            "loaded_count": loaded_count,
        }
    except Exception as e:
        logger.error(f"获取模型列表时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")
