# 部署指南

完整部署流程见根 [README.md](../README.md)。本文聚焦在生产部署的细节。

## 前置要求

- Docker ≥ 24
- Docker Compose v2(随 Docker Desktop 自带,Linux 用 `docker compose` 命令)
- NVIDIA Container Toolkit(GPU 子服务必需)

```bash
# Ubuntu / Debian 安装 NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 构建期 HTTP 代理

国内环境拉 PyPI / HuggingFace 慢,推荐配置代理:

```bash
# .env
HTTP_PROXY=http://host.docker.internal:7890
HTTPS_PROXY=http://host.docker.internal:7890
```

- macOS / Windows Docker Desktop:`host.docker.internal` 直接可用
- Linux 服务器:把 `host.docker.internal` 换成宿主机 LAN IP(例如 `192.168.1.10`),
  或在 docker daemon 里配 default proxy

代理仅用于构建期,Dockerfile 末尾会清空运行期 ENV。

## 启动方式

```bash
docker compose build              # 首次或 Dockerfile 变更后
docker compose up -d              # 默认: gateway + funasr + cosyvoice

# 加 dolphin
docker compose --profile dolphin up -d

# 加 qwen3-asr-vllm
docker compose --profile qwen3-asr up -d

# 全部启动
docker compose --profile dolphin --profile qwen3-asr up -d
```

## 多 GPU / 多副本

每个子服务一个容器 = 一张 GPU。多副本两步:

1. 在 `docker-compose.yml` 加新服务条目(`funasr-1`),把 `CUDA_VISIBLE_DEVICES` 设成另一张卡
2. 在网关 env 里把对应 URL 列表用逗号分隔:
   ```
   FUNASR_SERVICE_URLS=http://funasr-0:8001,http://funasr-1:8001
   ```

网关侧的 `_HttpReplicaPool` 会做最少连接 + 随机选副本调度。

注意:**cosyvoice 多副本的音色 CRUD 必须 sticky 到固定 primary 副本**,否则
`spk2info.pt` 写操作会冲突。当前没有自动 sticky 逻辑,生产多副本需要 LB 加规则
或者只让一个副本接收 `POST/DELETE /voices/*`。

## 数据卷

| 主机路径 | 容器路径 | 用途 | 注意 |
|---|---|---|---|
| `${MODELSCOPE_CACHE}` (默认 `~/.cache/modelscope`) | `/root/.cache/modelscope` | 模型权重缓存,所有 GPU 子服务共享 | 提前下载可省首次启动等待 |
| `./voices` | `/app/voices`(只在 cosyvoice) | 零样本克隆音色 + spk2info.pt | 持久化用户数据 |
| `./temp` | `/app/temp`(只在 gateway) | 网关临时音频文件 | |
| `./data` | `/app/data`(只在 gateway) | 异步 TTS 任务库 | |
| `./logs` | `/app/logs`(只在 gateway) | 日志 | |

## 健康检查与重启

每个子服务都有 `/health`,docker-compose 配了 `healthcheck`:
- funasr / dolphin: `start_period=180s`,等模型加载
- qwen3-asr / cosyvoice: `start_period=300s`,vLLM 与 CosyVoice 加载更慢

`gateway.depends_on` 用 `condition: service_started`,即使子服务还没就绪
网关也能起来对外报 503,避免一处异常全栈不可用。改 `service_healthy`
即可严格依赖。

## 鉴权

- **外部鉴权**(可选):设置 `APPTOKEN` / `APPKEY`,客户端通过
  `X-NLS-Token` 头(Aliyun 接口)或 `Authorization: Bearer xxx`(OpenAI 接口)携带
- **内部鉴权**(必备):`INTERNAL_SERVICE_TOKEN` 网关→子服务通过
  `X-Internal-Token` 头携带。生产环境务必改默认值

## 排错

```bash
# 看子服务状态
docker compose ps

# 看子服务日志
docker compose logs -f funasr-0

# 进容器查环境
docker compose exec gateway env | grep -E "SERVICE_URLS|TOKEN"

# 网关侧能不能连到子服务
docker compose exec gateway curl -fsS http://funasr-0:8001/health

# 重启某个服务(不影响其它)
docker compose restart funasr-0

# 完整清理(注意会丢容器,保留卷)
docker compose down

# 连卷一起清(会丢音色数据!)
docker compose down -v
```
