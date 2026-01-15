# AI Recorder Agent

> 🎙️ **实时录音 → Whisper ASR → 字幕级稳定输出 → LLM 修复 → 会后总结 & QA（RAG）**  
> 一个可本地部署、可扩展、面向 Agent 与硬件集成的语音智能后端系统。

---

## 一、项目简介（What & Why）

**AI Recorder Agent** 是一个以 **实时语音理解** 为核心的后端系统，目标是替代市面上“录音卡片 / 会议助手”的封闭方案，提供：

- ✅ **实时录音 + 分片上传（chunk）**
- ✅ **Whisper（faster-whisper）实时转写**
- ✅ **字幕级体验**
  - 实时滚动（partial）
  - 句子稳定落地（final）
  - 跨 chunk 去重
- ✅ **LLM 严格保真修复层**
  - 自动补标点
  - 专名规范
  - 去口癖
  - 不改语义、不编造
- ✅ **会后总结 + QA（RAG）**
  - 基于转写内容回答问题
  - 所有结论带时间戳引用
- ✅ **完全后端化**
  - 无 UI 依赖
  - Windows / macOS / Linux 客户端均可接入
  - 方便后续接硬件、Web UI、App

系统设计遵循两个原则：

> **可解释（每句话有来源）**  
> **可控（任何 LLM 输出都不“编造”）**

---

## 二、整体架构概览

```text
┌──────────────────┐
│ Mic / Audio File │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  Chunk Client    │
│  (Realtime /     │
│   Offline)       │
└─────────┬────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│            FastAPI Backend                │
│                                          │
│  ├─ Session 管理                          │
│  ├─ Chunk 接收 / 时间轴对齐               │
│  ├─ Whisper ASR (faster-whisper)          │
│  ├─ 字幕级去重与稳定输出 (partial/final) │
│  ├─ Transcript 持久化存储                 │
│  ├─ LLM Post-Edit（严格保真）             │
│  └─ RAG（Summary / QA，带时间戳引用）     │
│                                          │
└──────────────────────────────────────────┘
```

---

## 三、目录结构说明（重要）

```text
ai-recorder-agent/
├── app/
│   ├── main.py                  # FastAPI 入口
│   │
│   ├── api/
│   │   └── routes.py             # 所有 HTTP API（核心）
│   │
│   ├── asr/
│   │   └── transcriber.py        # faster-whisper 封装
│   │
│   ├── llm/
│   │   ├── postedit.py           # LLM 严格保真修复层
│   │   ├── prompts.py            # Post-edit Prompt
│   │   └── providers/
│   │       ├── base.py            # LLM Provider 抽象
│   │       └── openai_compat.py   # OpenAI / DeepSeek / 通义 兼容接口
│   │
│   ├── rag/
│   │   ├── index.py              # BM25 索引
│   │   ├── retriever.py          # 检索逻辑
│   │   ├── qa.py                 # 基于引用的 QA
│   │   └── summary.py            # 会后总结（带时间戳）
│   │
│   └── storage/
│       ├── session_store.py      # Session 目录与锁
│       ├── session_state.py      # 字幕级状态（partial/final）
│       └── artifacts.py          # clean transcript / summary 等产物
│
├── data/
│   └── sessions/                 # 运行时数据（git 忽略）
│
├── scripts/
│   └── mic_stream_client.py      # Windows 实时麦克风客户端示例
│
├── .env.example                  # 环境变量示例
├── .gitignore
├── pyproject.toml
└── README.md
```


---

## 四、运行环境要求

### 1️⃣ 后端（推荐：Linux / WSL）

- Python **3.10+**
- NVIDIA GPU（可选，但强烈推荐）
- CUDA 12.x（与驱动匹配）
- `ffmpeg` / `ffprobe`

### 2️⃣ 客户端

- Windows / macOS / Linux
- Python 3.x
- 麦克风（实时模式）

---

## 五、快速开始（10 分钟跑起来）

### Step 1：克隆项目

```bash
git clone https://github.com/<YOUR_NAME>/ai-recorder-agent.git
cd ai-recorder-agent
```

### Step 2: 创建 Python 环境（使用 uv）

```bash
pip install uv
uv venv
source .venv/bin/activate
uv add fastapi uvicorn httpx pydantic rank-bm25 soundfile python-multipart
```

### Step 3: Step 3：配置 LLM

修改.env中变量

### Step 4: 启动后端服务

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 六、核心使用方式

### 创建会话

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -F "model=small" \
  -F "language=zh"
````

返回示例：

```json
{
  "session_id": "xxxx",
  "model": "small",
  "language": "zh"
}
```

---

### 上传音频 chunk（实时 / 离线）

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/chunks \
  -F "file=@chunk0.wav"
```

返回内容包含：

* 转写 `segments`
* 对应的时间戳（绝对时间轴）

---

### 获取完整转写

```bash
curl http://127.0.0.1:8000/sessions/<SESSION_ID>/transcript
```

返回该 session 下当前累计的全部转写结果。

---

### LLM 修复（标点 / 专名 / 去口癖）

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/llm/postedit \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 40}'
```

修复后的产物保存于：

```text
data/sessions/<SESSION_ID>/artifacts/clean_transcript.jsonl
```

每一行对应一个 **严格保真** 的修复后 segment，并保留原始时间戳。

---

### 会后总结

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/summary \
  -H "Content-Type: application/json" \
  -d '{"guidance":"聚焦结论与行动项"}'
```

返回内容包含：

* 总结标题
* 关键要点
* 行动项（如有）
* 每一条结论对应的时间戳引用

---

### QA（RAG）

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"刚才的关键结论是什么？"}'
```

返回结果包含：

* 问题对应的答案
* 每一条结论所引用的原始转写时间戳（可回溯）

---

## 七、实时录音（Windows 示例）

在 Windows 端运行麦克风实时客户端：

```shell
python mic_stream_client.py \
  --base-url http://localhost:8000 \
  --device 0 \
  --language zh
```

字幕级输出示例：

```shell
[PARTIAL] 我们今天主要讨论
[FINAL] 我们今天主要讨论模型延迟问题。
```

说明：

* `PARTIAL`：实时滚动字幕（可能被后续修正）
* `FINAL`：已稳定落地的句子，不再修改
