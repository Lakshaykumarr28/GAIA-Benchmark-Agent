# GAIA Benchmark Agent

**A powerful multi-tool agent for the Hugging Face Agents Course GAIA Benchmark**

This repository contains my submission for the **Hugging Face Agents Course** final assignment. The agent is designed to tackle the challenging GAIA benchmark questions, which test an AI agent's ability to use tools, reason accurately, handle different file formats, and retrieve information reliably.

![GAIA Benchmark](https://img.shields.io/badge/GAIA-Benchmark-blue)
![Framework](https://img.shields.io/badge/Framework-LangGraph-orange)
![Model](https://img.shields.io/badge/Model-GPT--OSS--20B-green)

## Overview

This agent combines **LangGraph** for reliable agentic workflows with a rich set of specialized tools to handle:
- Mathematical calculations
- Web and Wikipedia searches
- Document processing (PDF, Excel, CSV, Word)
- Vision (OCR on images)
- Audio transcription
- YouTube video content understanding

The system is optimized for accuracy, tool selection intelligence, and robustness — key requirements for performing well on the GAIA benchmark.

## Key Features

- **Dynamic Tool Selection**: Automatically selects only relevant tools based on the question
- **Multi-Modal Capabilities**: Handles text, images, PDFs, spreadsheets, audio, and video transcripts
- **Reversed Text Detection**: Special handling for GAIA's tricky reversed-string questions
- **Strict Output Formatting**: Enforces `FINAL ANSWER: <answer>` format required by the benchmark
- **Verbose Debugging**: Excellent logging for development and transparency
- **Gradio Interface**: Easy-to-use web UI for testing

## Directory Structure

```bash
GAIA-Benchmark-Agent/
├── agent.py                 # Core agent logic, tools, and graph
├── app.py                   # Gradio UI + HF Space integration
├── system_prompt.txt        # Strict system instructions for GAIA
├── requirements.txt         # All dependencies
├── README.md                # This file
├── Hugging face Agents Course.pdf  # Completion Certificate
├── __pycache__/             # Python cache
└── .env.example             # (Optional) Environment variables template
```

### File Explanations

- **`agent.py`**: The heart of the project. Contains all custom tools, LangGraph workflow, model configuration, and the `run_agent()` function.
- **`app.py`**: Hugging Face Space compatible Gradio interface that connects to the official GAIA evaluation endpoint.
- **`system_prompt.txt`**: Carefully crafted instructions emphasizing factual accuracy, tool usage rules, and strict output format.
- **`requirements.txt`**: Comprehensive list of packages for document handling, OCR, audio, LLMs, etc.

## Framework Used

**LangGraph** (part of LangChain ecosystem) was chosen as the primary framework because:

- It provides **stateful, controllable workflows** with cycles (agent → tools → agent)
- Excellent **tool calling** support with `bind_tools()`
- Built-in **conditional routing** (`tools_condition`)
- Better debugging and observability than simple ReAct agents
- Production-grade reliability

The agent uses a **ReAct-style** loop with a custom `StateGraph`.

## Models Tested

I experimented with several strong open-source and hosted models:

| Model                  | Size   | Performance Notes                     | Used? |
|------------------------|--------|---------------------------------------|-------|
| **GPT-OSS-20B**        | 20B    | **Best balance** of speed & capability | Yes   |
| Qwen3-32B              | 32B    | Strong reasoning but slower           | Tested |
| Llama 3.1 8B           | 8B     | Fast but weaker on complex tool use   | Tested |

**Conclusion**: `openai/gpt-oss-20b` (via Groq) was the best performer for this benchmark — offering excellent tool-calling accuracy and reasoning while remaining fast enough for the evaluation.

The model is configured with:
- `temperature=0` (for maximum determinism)
- Tool binding with automatic tool choice

## Tools Developed

I created **9 specialized tools**:

### Core Tools
1. **`calculator`** - Safe mathematical expression evaluation
2. **`web_search`** - Tavily-powered web search (used only when necessary)
3. **`wiki_search`** - Wikipedia lookup with summaries

### Document & File Tools
4. **`read_pdf`** - Extract text from local PDFs
5. **`read_excel`** - Read Excel spreadsheets
6. **`read_csv`** - Read CSV files
7. **`read_word`** - Extract text from `.docx` files

### Multi-Modal Tools
8. **`ocr_image`** - Tesseract + PIL based OCR for images
9. **`speech_to_text`** - Whisper-based audio transcription
10. **`get_youtube_transcript`** - Extract transcripts from YouTube videos

**Smart Tool Selection Logic** is implemented in `get_tools()` — the agent only loads tools relevant to each specific question.

## System Prompt Strategy

The `system_prompt.txt` is very strict:
- Emphasizes **no hallucination**
- Forces tool usage for verification
- Requires exact `FINAL ANSWER:` format
- Special handling for YouTube URLs
- Rules for when *not* to use tools

## How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Lakshaykumarr28/GAIA-Benchmark-Agent.git
   cd GAIA-Benchmark-Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (`.env`):
   ```env
   GROQ_API_KEY=your_groq_key
   TAVILY_API_KEY=your_tavily_key
   HF_TOKEN=your_hf_token
   ```

4. Run the Gradio app:
   ```bash
   python app.py
   ```

## Hugging Face Space

This repository is also deployed as a Hugging Face Space with OAuth integration for the official GAIA evaluation.

