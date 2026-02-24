# LLM Gateway

统一的 LLM 接口代理，将本地和远程 LLM API 标准转换为 OpenAI 兼容的 HTTP 端点。

## 功能特性

- **OpenAI 兼容 API**: OpenAI API 的直接替代品
- **多提供商支持**: OpenAI、Anthropic、Ollama、Azure 等
- **流式响应支持**: 实时流式输出聊天完成内容
- **分布式架构**: Controller-Worker 模式，支持水平扩展
- **智能路由**: 基于 Least-Loaded 的负载均衡 + 熔断器
- **健康监控**: 自动 Worker 健康检测与故障转移
- **指标监控**: Prometheus 兼容指标
- **错误处理**: 跨提供商的统一错误响应

## 部署模式

LLM Gateway 支持两种部署模式：

### 1. 单体模式（单进程）

所有适配器运行在单个进程中。简单但局限于单台机器。

```bash
llm-gateway server --port 8000
```

### 2. Controller-Worker 模式（分布式）

分布式架构，Controller 和 Worker 进程分离。

```
┌─────────────────────────────────────────────────────────────┐
│                      Controller (:8000)                       │
│  • 管理 Worker 注册表                                         │
│  • 使用 Least-Loaded 策略路由请求                             │
│  • 健康监控与熔断器                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│   Worker (:8001)        │ │   Worker (:8002)        │
│  • 向 Controller 注册    │ │  • 向 Controller 注册    │
│  • 代理请求到后端       │ │  • 代理请求到后端       │
└─────────────────────────┘ └─────────────────────────┘
```

## 快速开始

### 安装

使用 **uv**（推荐）：

```bash
# 安装 uv（如果尚未安装）
pip install uv

# 安装项目依赖
uv pip install -e ".[dev]"
```

或使用 pip：

```bash
pip install -e ".[dev]"
```

### 单体模式

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
OLLAMA_HOST=http://localhost:11434
```

启动服务：

```bash
llm-gateway server --port 8000
```

### Controller-Worker 模式

#### 1. 启动 Controller

```bash
llm-gateway controller --port 8000
```

可选配置：

```env
# Controller 设置
CONTROLLER_HOST=0.0.0.0
CONTROLLER_PORT=8000
INTERNAL_API_KEY=your-secret-key  # 可选：保护内部端点
HEARTBEAT_TIMEOUT=60              # Worker 被认为失联的秒数
```

#### 2. 启动 Worker

每个 Worker 服务于单个模型实例：

```bash
# 第一个 Ollama 实例的 llama3 Worker
llm-gateway worker \
  --worker-id llama3-gpu-001 \
  --model-id llama3 \
  --backend-url http://localhost:11434 \
  --controller-url http://localhost:8000 \
  --port 8001

# 第二个 Ollama 实例的 llama3 Worker（负载均衡）
llm-gateway worker \
  --worker-id llama3-gpu-002 \
  --model-id llama3 \
  --backend-url http://localhost:11435 \
  --controller-url http://localhost:8000 \
  --port 8002

# mistral Worker
llm-gateway worker \
  --worker-id mistral-cpu-001 \
  --model-id mistral \
  --backend-url http://localhost:11436 \
  --controller-url http://localhost:8000 \
  --port 8003
```

Worker 配置：

```env
# Worker 设置
WORKER_ID=llama3-gpu-001          # 必填：唯一标识符
MODEL_ID=llama3                    # 必填：此 Worker 服务的模型名称
CONTROLLER_URL=http://localhost:8000
BACKEND_URL=http://localhost:11434  # 必填：实际后端服务（Ollama/OpenAI 等）
LISTEN_PORT=8001
CAPACITY=10                       # 最大并发请求数
HEARTBEAT_INTERVAL=10             # 心跳频率（秒）
```

## 使用方法

### 列出可用模型

```bash
curl http://localhost:8000/v1/models
```

### 聊天完成

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

### 流式响应

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": true
  }'
```

### 内部 API（Worker 管理）

```bash
# 列出所有 Worker
curl http://localhost:8000/internal/workers

# 按模型筛选
curl "http://localhost:8000/internal/workers?model_id=llama3"
```

## 架构

### 单体模式

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   客户端    │────▶│   Gateway    │────▶│    适配器       │
│  (OpenAI    │     │  (FastAPI)   │     │ (OpenAI/Claude/ │
│  格式)      │◄────│              │◀────│  Ollama 等)     │
└─────────────┘     └──────────────┘     └─────────────────┘
```

### Controller-Worker 模式

```
┌─────────────────────────────────────────────────────────────┐
│                        Controller                            │
├─────────────────────────────────────────────────────────────┤
│  公开端点:   POST /v1/chat/completions                     │
│              GET  /v1/models                                │
│  内部端点:   POST /internal/workers/register               │
│              POST /internal/workers/heartbeat              │
│              GET  /internal/workers                         │
├─────────────────────────────────────────────────────────────┤
│  • Worker 注册表 (按模型、 按 ID)                            │
│  • 路由 (Least-Loaded + 熔断器)                              │
│  • 健康检查 (心跳超时 + 主动探测)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                    HTTP（透明代理）
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Worker                               │
├─────────────────────────────────────────────────────────────┤
│  • 注册客户端 (自动注册 + 心跳)                                │
│  • 代理处理器 (转发请求到后端)                                 │
│  • 负载追踪 (请求时递增/递减)                                  │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 说明 |
|------|------|
| **Worker 注册表** | 维护 model_id → workers 映射 |
| **路由** | 使用 Least-Loaded 策略选择 Worker |
| **熔断器** | 防止级联故障，自动恢复 |
| **健康检查器** | 监控心跳超时，探测 Worker 状态 |
| **注册客户端** | Worker 端：注册 + 发送心跳 |
| **代理处理器** | Worker 端：转发请求到后端 |

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 服务主机（单体模式） |
| `PORT` | `8000` | 服务端口（单体模式） |
| `CONTROLLER_HOST` | `0.0.0.0` | Controller 主机 |
| `CONTROLLER_PORT` | `8000` | Controller 端口 |
| `INTERNAL_API_KEY` | - | 内部端点 API 密钥 |
| `HEARTBEAT_TIMEOUT` | `60` | Worker 被认为失联的秒数 |
| `HEARTBEAT_CHECK_INTERVAL` | `10` | 健康检查间隔 |
| `WORKER_ID` | - | Worker 唯一标识符 |
| `MODEL_ID` | - | 此 Worker 服务的模型 |
| `CONTROLLER_URL` | `http://localhost:8000` | Controller URL |
| `BACKEND_URL` | - | 后端服务 URL |
| `LISTEN_PORT` | `8001` | Worker 监听端口 |
| `CAPACITY` | `10` | 最大并发请求数 |
| `HEARTBEAT_INTERVAL` | `10` | 心跳频率（秒） |
| `REGISTRY_RETRY_COUNT` | `30` | 注册重试次数 |
| `REGISTRY_RETRY_DELAY` | `5` | 初始重试延迟（秒） |

## 开发

详细开发指南请参阅 [AGENTS.md](AGENTS.md)。

### 运行测试

```bash
pytest
```

### 代码质量

```bash
ruff check . && ruff format . && mypy llm_gateway
```

## 许可证

MIT
