# TODO

## 环境变量语义重定义

### `ASR_MODEL_MODE` → `FUNASR_MODEL_MODE`

**问题**: 当前 `ASR_MODEL_MODE` 命名暗示它是全局 ASR 模式,但实际只有 funasr 子服务读它 (`services/funasr/server.py:52-350`)。qwen3-asr 和 dolphin 完全不读 — qwen3-asr 同一份 1.7B 模型同时处理 offline/realtime,dolphin 只有 offline。

**影响**: gateway 仍将其注入 `list_models` 响应和兼容性校验 (`app/services/asr/manager.py:215`),当默认 ASR 引擎是 qwen3-asr 时,这个字段对客户端是误导。

**计划**:
- [ ] `ASR_MODEL_MODE` → `FUNASR_MODEL_MODE`,仅保留在 funasr-0 服务块 (`docker-compose.yml` + `services/funasr/server.py`)
- [ ] gateway 侧彻底移除对 `ASR_MODEL_MODE` 的读取和注入 (`app/core/config.py:43`, `app/services/asr/manager.py:122/206/215`)
- [ ] gateway `list_models` 的 `asr_model_mode` 字段替换为 per-engine 信息 (或直接删掉)
- [ ] 更新 `docs/deployment.md` §6.1 和 `README.md` 服务列表

### `TTS_MODEL_MODE` → `COSYVOICE_MODEL_MODE`

**问题**: 命名相比 ASR 端好一些 (只有一个 TTS 引擎),但仍与 `FUNASR_MODEL_MODE` 风格不一致。

**计划**:
- [ ] `TTS_MODEL_MODE` → `COSYVOICE_MODEL_MODE`,docker-compose + cosyvoice server + gateway 同步改
- [ ] gateway 的 `get_voices()` / `get_voices_info()` 过滤逻辑保持不变 (`app/services/tts/http_engine.py:292`)
- [ ] 更新 `docs/deployment.md` §6.4 和 `README.md`

### 迁移兼容

两个改名都是 **break 现有 .env 文件**,需要:
- [ ] 在 gateway 启动时检测旧变量名并 warn (不报错,给一个版本缓冲)
- [ ] 更新 `scripts/plan_deployment.py` 输出的 env 片段用新变量名
- [ ] `docs/migration_to_latest.md` 加迁移说明