---
title: Multi-Source RAG AI Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---
 

# 🤖 AI Chat Assistant

[![CI](https://github.com/kumardatascience/ai-chat-assistant-themis/actions/workflows/ci.yml/badge.svg)](https://github.com/kumardatascience/ai-chat-assistant-themis/actions/workflows/ci.yml)

An intelligent AI chatbot that routes queries between direct LLM, document retrieval (RAG), and live web search — built with **LlamaIndex Workflows**, **Gemini**, **ChromaDB**, **Tavily**, and **Chainlit**.

## ✨ Features

- 💬 ChatGPT-like UI with streaming responses
- 🚦 Intelligent query routing (direct / rag / web / multi)
- 📚 Document Q&A with ChromaDB
- 🌐 Live web search via Tavily
- 🧠 Conversation memory
- 🔍 Transparent reasoning steps
- 🐳 Dockerized
- ✅ CI tested via GitHub Actions

## 🚀 Quick Start

**Prerequisites:** Python 3.12+, [Gemini API key](https://aistudio.google.com/apikey), [Tavily API key](https://tavily.com)

```bash
# Clone and set up
git clone https://github.com/kumardatascience/ai-chat-assistant-themis.git
cd ai-chat-assistant-themis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env and fill in GOOGLE_API_KEY and TAVILY_API_KEY

# (Optional) Add documents to data/documents/ then index them
python3 src/app/rag.py

# Run
cd src/app
chainlit run main.py -w
```

Open [http://localhost:8000](http://localhost:8000).

## 🐳 Docker

```bash
docker build -t ai-chat-assistant .
docker run --rm -p 8000:8000 --env-file .env ai-chat-assistant
```

## 🧪 Tests

```bash
pytest -v
```

## 📁 Project Structure

```
src/app/
├── main.py          # Chainlit entry point
├── llm.py           # Gemini wrapper
├── rag.py           # Document indexing + retrieval
├── web_search.py    # Tavily search
└── router.py        # LlamaIndex Workflow router
data/documents/      # Your PDFs/text files
tests/               # Pytest tests
Dockerfile
.github/workflows/ci.yml
```

## 🛠️ Tech Stack

Chainlit · Google Gemini · LlamaIndex Workflows · ChromaDB · Tavily · HuggingFace embeddings (BAAI/bge-small-en-v1.5) · pytest · Docker · GitHub Actions

## 📝 License

MIT
