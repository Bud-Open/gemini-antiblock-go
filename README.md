# Gemini Antiblock Proxy (Go 版本)

这是一个用 Go 语言重写的 Gemini API 代理服务器，具有强大的流式重试和标准化错误响应功能。它可以处理模型的"思考"过程，并在重试后过滤思考内容以保持干净的输出流。

## 功能特性

- **流式响应处理**: 支持 Server-Sent Events (SSE)流式响应
- **智能重试机制**: 当流被中断时自动重试，最多支持 100 次连续重试
- **思考内容过滤**: 可以在重试后过滤模型的思考过程，保持输出的整洁
- **标准化错误响应**: 提供符合 Google API 标准的错误响应格式
- **CORS 支持**: 完整的跨域资源共享支持
- **环境变量配置**: 通过环境变量进行灵活配置
- **详细日志记录**: 支持调试模式和详细的操作日志

## 安装和运行

### 方法一：使用 Docker（推荐）

#### 使用预构建的镜像

```bash
# 从 GitHub Container Registry 拉取镜像
docker pull ghcr.io/davidasx/gemini-antiblock-go:latest

# 运行容器
docker run -d \
  --name gemini-antiblock \
  -p 8080:8080 \
  -e UPSTREAM_URL_BASE=https://generativelanguage.googleapis.com \
  -e MAX_CONSECUTIVE_RETRIES=100 \
  -e DEBUG_MODE=false \
  -e RETRY_DELAY_MS=750 \
  -e SWALLOW_THOUGHTS_AFTER_RETRY=true \
  ghcr.io/davidasx/gemini-antiblock-go:latest
```

#### 使用 Docker Compose

```bash
# 克隆仓库
git clone https://github.com/Davidasx/gemini-antiblock-go.git
cd gemini-antiblock-go

# 使用本地构建
docker-compose up -d

# 或使用预构建镜像
docker-compose --profile prebuilt up -d gemini-antiblock-prebuilt
```

#### 本地构建 Docker 镜像

```bash
# 构建镜像
docker build -t gemini-antiblock-go .

# 运行容器
docker run -d \
  --name gemini-antiblock \
  -p 8080:8080 \
  -e DEBUG_MODE=false \
  gemini-antiblock-go
```

### 方法二：从源码运行

#### 前置要求

- Go 1.21 或更高版本

#### 安装依赖

```bash
go mod download
```

### 配置

1. 复制环境变量示例文件：

```bash
cp .env.example .env
```

2. 编辑 `.env` 文件配置你的设置：

```bash
# Gemini API基础URL
UPSTREAM_URL_BASE=https://generativelanguage.googleapis.com

# 最大连续重试次数
MAX_CONSECUTIVE_RETRIES=100

# 启用调试模式
DEBUG_MODE=false

# 重试延迟（毫秒）
RETRY_DELAY_MS=750

# 重试后是否吞掉思考内容
SWALLOW_THOUGHTS_AFTER_RETRY=true

# 服务器端口
PORT=8080

# 速率限制
ENABLE_RATE_LIMIT=false
# 速率限制窗口时间（秒）
RATE_LIMIT_WINDOW_SECONDS=60
# 速率限制请求数
RATE_LIMIT_COUNT=10

# 是否启用实验性的“跨尝试句末标点”启发式
ENABLE_PUNCTUATION_HEURISTIC=true
```

### 运行

```bash
go run main.go
```

或者编译后运行：

```bash
go build -o gemini-antiblock
./gemini-antiblock
```

服务器将在指定端口启动（默认 8080）。

## 环境变量配置

| 变量名                         | 默认值                                      | 描述                                                                                      |
| ------------------------------ | ------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `UPSTREAM_URL_BASE`            | `https://generativelanguage.googleapis.com` | Gemini API 的基础 URL                                                                     |
| `MAX_CONSECUTIVE_RETRIES`      | `100`                                       | 流中断时的最大连续重试次数                                                                |
| `DEBUG_MODE`                   | `true`                                      | 是否启用调试日志                                                                          |
| `RETRY_DELAY_MS`               | `750`                                       | 重试间隔时间（毫秒）                                                                      |
| `SWALLOW_THOUGHTS_AFTER_RETRY` | `true`                                      | 重试后是否过滤思考内容                                                                    |
| `PORT`                         | `8080`                                      | 服务器监听端口                                                                            |
| `ENABLE_RATE_LIMIT`            | `false`                                     | 是否启用速率限制                                                                          |
| `RATE_LIMIT_COUNT`             | `10`                                        | 速率限制请求数                                                                            |
| `RATE_LIMIT_WINDOW_SECONDS`    | `60`                                        | 速率限制窗口时间（秒）                                                                    |
| `ENABLE_PUNCTUATION_HEURISTIC` | `true`                                      | 启用实验性的“跨尝试句末标点”启发式，用于在连续 3 次续上尝试都以句末标点结尾时提前判定成功 |

## 使用方法

代理服务器启动后，你可以将 Gemini API 的请求发送到这个代理服务器。代理会自动：

1. 转发请求到上游 Gemini API
2. 处理流式响应
3. 在流中断时自动重试
4. 注入系统提示确保响应以`[done]`结尾
5. 过滤重试后的思考内容（如果启用）

### 示例请求

```bash
curl "http://127.0.0.1:8080/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse" \
   -H "x-goog-api-key: $GEMINI_API_KEY" \
   -H 'Content-Type: application/json' \
   -X POST --no-buffer  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {
            "text": "Hello"
          }
        ]
      }
    ],
    "generationConfig": {
      "thinkingConfig": {
        "includeThoughts": true
      }
    }
  }'
```

## 项目结构

```
gemini-antiblock-go/
├── main.go                 # 主程序入口
├── config/
│   └── config.go          # 配置管理
├── logger/
│   └── logger.go          # 日志记录
├── handlers/
│   ├── errors.go          # 错误处理和CORS
│   ├── health.go          # 健康检查
│   └── proxy.go           # 代理处理逻辑
├── streaming/
│   ├── sse.go             # SSE流处理
│   └── retry.go           # 重试逻辑
├── mock-server/           # 测试模拟服务器
│   ├── main.go            # 模拟服务器入口
│   ├── go.mod             # 模拟服务器依赖
│   └── README.md          # 模拟服务器文档
├── go.mod                 # Go模块文件
├── go.sum                 # 依赖校验和
├── .env.example           # 环境变量示例
├── Dockerfile             # Docker构建文件
├── docker-compose.yml     # Docker Compose配置
├── DEPLOYMENT.md          # 部署文档
├── LICENSE                # 许可证文件
└── README.md              # 项目文档
```

## 重试机制

当检测到以下情况时，代理会自动重试：

1. **流中断**: 流意外结束而没有完成标记
2. **内容被阻止**: 检测到内容被过滤或阻止
3. **思考中完成**: 在思考块中检测到完成标记（无效状态）
4. **异常完成原因**: 非正常的完成原因
5. **不完整响应**: 响应看起来不完整

重试时会：

- 保留已生成的文本作为上下文
- 构建继续对话的新请求
- 在达到最大重试次数后返回错误

## 日志记录

代理提供三个级别的日志：

- **DEBUG**: 详细的调试信息（仅在调试模式下显示）
- **INFO**: 一般信息和操作状态
- **ERROR**: 错误信息和异常

## 测试和开发

### Mock Server

项目包含一个用于测试的模拟服务器（`mock-server/`），提供多种测试场景来验证代理服务器的行为：

#### 功能特性

- **多种测试用例**: 支持 3 种不同的测试场景
- **路径路由**: 使用 `/type-1`, `/type-2`, `/type-3` 路径来区分测试用例
- **随机延迟**: 模拟真实 API 的响应延迟（50-200ms）
- **流式响应**: 支持 Server-Sent Events (SSE)格式
- **思考内容**: 模拟包含思考过程的响应
- **标准响应格式**: 包含 `finishReason: "STOP"` 参数

#### 测试用例

| 测试用例 | 路径      | 描述                                      | 用途                   |
| -------- | --------- | ----------------------------------------- | ---------------------- |
| 1        | `/type-1` | 包含思考内容，但不返回 `[done]` 标记      | 测试处理不完整流的能力 |
| 2        | `/type-2` | 将 `[done]` 标记分割发送（`[do` + `ne]`） | 测试处理分割标记的能力 |
| 3        | `/type-3` | 返回空响应                                | 测试处理空响应的能力   |

#### 使用方法

1. **启动模拟服务器**：

   ```bash
   cd mock-server
   go run main.go
   ```

   服务器将在端口 8081 启动。

2. **直接测试**：

   ```bash
   # 测试用例 1
   curl -X POST "http://localhost:8081/type-1/v1beta/models/gemini-pro:streamGenerateContent" \
     -H "Content-Type: application/json" \
     -d '{"contents": [{"parts": [{"text": "Hello"}]}]}'
   ```

3. **与代理服务器配合测试**：

   ```bash
   # 配置代理指向不同测试用例
   UPSTREAM_URL_BASE=http://localhost:8081/type-1 go run main.go  # 测试用例1
   UPSTREAM_URL_BASE=http://localhost:8081/type-2 go run main.go  # 测试用例2
   UPSTREAM_URL_BASE=http://localhost:8081/type-3 go run main.go  # 测试用例3

   # 然后通过代理发送请求
   curl -X POST "http://localhost:8080/v1beta/models/gemini-pro:streamGenerateContent" \
     -H "Content-Type: application/json" \
     -d '{"contents": [{"parts": [{"text": "Test message"}]}]}'
   ```

4. **健康检查**：
   ```bash
   curl http://localhost:8081/health
   ```

详细的使用说明请参考 [`mock-server/README.md`](mock-server/README.md)。

## Docker 部署

### 环境变量

在 Docker 容器中可以通过环境变量配置应用：

```bash
docker run -d \
  --name gemini-antiblock \
  -p 8080:8080 \
  -e UPSTREAM_URL_BASE=https://generativelanguage.googleapis.com \
  -e MAX_CONSECUTIVE_RETRIES=100 \
  -e DEBUG_MODE=false \
  -e RETRY_DELAY_MS=750 \
  -e SWALLOW_THOUGHTS_AFTER_RETRY=true \
  -e ENABLE_PUNCTUATION_HEURISTIC=true \
  -e PORT=8080 \
  ghcr.io/davidasx/gemini-antiblock-go:latest
```

### 健康检查

容器包含内置的健康检查功能，会定期检查服务状态：

```bash
# 查看容器健康状态
docker ps

# 查看健康检查日志
docker inspect --format='{{json .State.Health}}' gemini-antiblock

# 手动执行健康检查
docker exec gemini-antiblock curl -f http://localhost:8080/
```

### 多架构支持

Docker 镜像支持多种架构：

- `linux/amd64` (x86_64)
- `linux/arm64` (ARM64)

Docker 会自动选择适合您系统架构的镜像。

### 生产部署建议

1. **使用特定版本标签**：避免使用 `latest` 标签，使用特定版本如 `v1.0.0`
2. **设置资源限制**：
   ```bash
   docker run -d \
     --name gemini-antiblock \
     --memory=256m \
     --cpus=0.5 \
     -p 8080:8080 \
     ghcr.io/davidasx/gemini-antiblock-go:v1.0.0
   ```
3. **使用 Docker Compose**：便于管理和扩展
4. **配置日志轮转**：避免日志文件过大
5. **设置重启策略**：确保服务高可用

## CI/CD

项目使用 GitHub Actions 自动构建和发布 Docker 镜像：

- **触发条件**：推送到 `main`/`master` 分支或创建标签
- **构建平台**：支持 `linux/amd64` 和 `linux/arm64`
- **发布位置**：`ghcr.io/davidasx/gemini-antiblock-go`
- **标签策略**：
  - `latest`：最新的 main 分支构建
  - `v1.0.0`：版本标签
  - `main`：main 分支构建

## 许可证

MIT License

## 原始版本

这是基于 Cloudflare Worker 版本的 Go 语言重写版本。原始 JavaScript 版本提供了相同的功能，但运行在 Cloudflare Workers 平台上。
