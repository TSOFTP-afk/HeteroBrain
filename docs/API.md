# VITA HTTP API 文档 / HTTP API Reference

> 本文档描述 VITA 引擎的 OpenAI 兼容 serve 模式对外暴露的 HTTP 接口。
> 实现参考：`src/vita/engine.cpp`（`run_serve()`），服务由 `src/vita/http_server.h`（Winsock2 单线程 HTTP/1.1）提供。

## 概览 / Overview

- **Base URL**：`http://127.0.0.1:<port>/v1`（默认端口 `8899`，`--port` 可改）
- **鉴权**：`Authorization: Bearer <api-key>`（`--api-key` 配置，默认 `thetrueai`）
- **传输**：HTTP/1.1，`Content-Length` 定长，`Connection: close`
- **编码**：请求/响应均为 UTF-8 JSON；响应头 `Content-Type: application/json; charset=utf-8`
- **行为**：每次 `chat/completions` 或 `world` 请求都会让 SNN 推进 `mod_update_interval`（默认 10）步，情感状态跨请求持续演化（服务进程内连续）

启动方式：

```powershell
vita_engine.exe --serve --port 8899 --api-key thetrueai --model-name thetrueai `
    --resume checkpoints/<ckpt>.snn2e --llm F:\hb_models\Qwen3-4B-Q4_K_M.gguf
```

## 鉴权 / Authentication

所有端点（除 `OPTIONS` 预检外）要求请求头：

```
Authorization: Bearer <api-key>
```

- Key 错误或缺失 → `401`，响应体 `{"error": "invalid_api_key", ...}`
- `OPTIONS` 预检请求直接放行（返回 204），便于浏览器/CORS 客户端探测

## 错误格式 / Error Format

所有错误响应均为 JSON，`status` 字段为 HTTP 状态码：

```json
{
  "error": "invalid_api_key",
  "status": 401,
  "message": "..."
}
```

| HTTP 状态码 | error | 场景 |
|---|---|---|
| 401 | `invalid_api_key` | 鉴权失败 |
| 400 | `bad_request` | 请求体非法（如未知事件类型、参数越界） |
| 404 | `not_found` | 路径不存在：`{"error":"not_found","message":"Not found: <METHOD> <PATH>"}` |

---

## 1. `GET /v1/models` — 列出模型 / List Models

返回当前引擎使用的模型名（与 `--model-name` 一致）。

**请求**：

```
GET /v1/models
Authorization: Bearer thetrueai
```

**响应** `200`：

```json
{
  "object": "list",
  "data": [
    { "id": "thetrueai", "object": "model", "created": 0, "owned_by": "thetrueai" }
  ]
}
```

---

## 2. `POST /v1/chat/completions` — 对话 / Chat Completion

OpenAI 兼容对话接口。引擎取**最新一条 user 消息**作为 SNN 输入，SNN 推进后读出情感状态，经情感调制（system 情感文字 + logit_bias + 采样参数）后由 LLM 生成回复。

**请求体**（OpenAI chat.completion 格式，仅 `messages` 必填）：

```json
{
  "model": "thetrueai",
  "messages": [
    { "role": "system", "content": "你是……" },
    { "role": "user", "content": "今天过得怎么样？" }
  ],
  "temperature": 0.8,
  "max_tokens": 768
}
```

**响应** `200`：

```json
{
  "id": "chatcmpl-<sn>",
  "object": "chat.completion",
  "model": "thetrueai",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "……（中文回复）" },
      "finish_reason": "stop"
    }
  ],
  "created": 1234567890
}
```

**说明**：
- `messages` 会以客户端提供的完整历史构建 prompt（与第三方软件历史一致），不依赖引擎内部历史
- `max_tokens` 上限 768（思考模式保留：`</think>` 片段 EOG 豁免）
- 情感状态随每次请求持续演化，同一进程内不重置

---

## 3. `POST /v1/world` — 世界事件注入 / World Event Injection

将外部世界事件注入 SNN（杏仁核 + 联合皮层直通通道），推进 `mod_update_interval` 步，返回当前皮质醇水平。这是引擎侧的"接口 A（inject_world）"落地。

**请求体**：

```json
{
  "type": "criticism",
  "intensity": 30
}
```

- `type`：事件类型字符串（见下表）；或 `event_type`：整数编号 `0..10`
- `intensity`：可选，`-50..50`，默认 `30`，越界自动 clamp

**响应** `200`：

```json
{
  "ok": 1,
  "event_type": 5,
  "type": "criticism",
  "intensity": 30,
  "cortisol": 0.7312
}
```

**错误**：`type`/`event_type` 无效 → `400` `{"error":"bad_request","message":"unknown event type ..."}`。

### 事件类型表 / Event Types

| 编号 | 字符串 | 含义 | 效价 |
|---|---|---|---|
| 0 | `food_tasty` | 食物美味 | 正性 |
| 1 | `food_bland` | 食物平淡 | 负性 |
| 2 | `threat_physical` | 身体威胁 | 负性 |
| 3 | `threat_social` | 社交威胁 | 负性 |
| 4 | `praise` | 表扬 | 正性 |
| 5 | `criticism` | 批评 | 负性 |
| 6 | `social_bond` | 社交联结 | 正性 |
| 7 | `social_loss` | 社交丧失 | 负性 |
| 8 | `achievement` | 成就达成 | 正性 |
| 9 | `novelty` | 新奇 | 中性 |
| 10 | `question` | 知识性问题 | 中性 |

---

## 示例 / Examples

### PowerShell（注意中文请求体须用 UTF-8 字节）

```powershell
# 对话
$body = [Text.Encoding]::UTF8.GetBytes('{"messages":[{"role":"user","content":"你好"}]}')
Invoke-RestMethod -Uri http://127.0.0.1:8899/v1/chat/completions `
    -Method Post -Headers @{Authorization='Bearer thetrueai'} -ContentType 'application/json' -Body $body

# 注入批评事件
$evt = [Text.Encoding]::UTF8.GetBytes('{"type":"criticism","intensity":40}')
Invoke-RestMethod -Uri http://127.0.0.1:8899/v1/world `
    -Method Post -Headers @{Authorization='Bearer thetrueai'} -ContentType 'application/json' -Body $evt
```

### curl

```bash
curl http://127.0.0.1:8899/v1/models -H "Authorization: Bearer thetrueai"

curl http://127.0.0.1:8899/v1/chat/completions \
  -H "Authorization: Bearer thetrueai" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'

curl http://127.0.0.1:8899/v1/world \
  -H "Authorization: Bearer thetrueai" -H "Content-Type: application/json" \
  -d '{"type":"achievement","intensity":50}'
```

---

## 兼容性说明 / Compatibility Notes

- 客户端配置：API 主机 `http://127.0.0.1:8899/v1`、API Key 默认 `thetrueai`、模型名默认 `thetrueai`（与 `--api-key`/`--model-name` 对应）
- `.NET` 客户端注意：响应头必须 `charset=utf-8`（已内置），否则中文按 Latin1 解码乱码
- 单线程 HTTP 服务：一次处理一个请求，不适合高并发；面向本地/单用户接入

## 与 OpenAI API 的差异 / Differences from OpenAI API

| 项 | OpenAI | VITA |
|---|---|---|
| 鉴权 | `Bearer sk-...` | `Bearer <任意 api-key>` |
| 请求推进 | 无状态 | SNN 每请求推进 10 步，情感跨请求演化 |
| 流式 `stream` | 支持 | 不支持（返回完整 JSON） |
| `/v1/embeddings` 等 | 有 | 无（仅 models / chat / world） |
| 工具调用 | 支持 | 不支持（w_tool=0，工具决策归 LLM） |
