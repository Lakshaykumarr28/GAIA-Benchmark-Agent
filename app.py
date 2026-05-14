import os
import re
import tempfile
import traceback

import gradio as gr
import requests
import pandas as pd

# ── Optional: load .env locally ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# =============================================================================
# TOOLS
# =============================================================================

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
            for r in results
        )
    except Exception as e:
        return f"[web_search error] {e}"


def visit_webpage(url: str) -> str:
    """Fetch and return the text content of a webpage."""
    try:
        import requests as req
        from markdownify import markdownify
        r = req.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = markdownify(r.text)
        # Truncate to avoid massive context
        return text[:8000]
    except Exception as e:
        return f"[visit_webpage error] {e}"


def download_task_file(task_id: str) -> str:
    """Download the file attached to a GAIA task. Returns text content or local file path."""
    url = f"{DEFAULT_API_URL}/files/{task_id}"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return f"[download_task_file error] {e}"

    content_disp = r.headers.get("Content-Disposition", "")
    content_type = r.headers.get("Content-Type", "")

    filename = ""
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[-1].strip().strip('"')
    ext = os.path.splitext(filename)[-1].lower() if filename else ""

    # Infer ext from MIME if missing
    if not ext:
        mime_map = {
            "csv": ".csv", "json": ".json", "text": ".txt",
            "mpeg": ".mp3", "audio": ".mp3",
            "png": ".png", "jpeg": ".jpg", "image": ".png",
            "excel": ".xlsx", "spreadsheet": ".xlsx",
            "pdf": ".pdf",
        }
        for key, val in mime_map.items():
            if key in content_type:
                ext = val
                break

    TEXT_EXTS = {".csv", ".txt", ".json", ".tsv", ".md", ".xml", ".html"}

    if ext in TEXT_EXTS:
        try:
            return r.content.decode("utf-8")
        except UnicodeDecodeError:
            return r.content.decode("latin-1")

    # Save binary to temp file
    suffix = ext or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(r.content)
        local_path = tmp.name

    return local_path  # Caller will route to the right tool


def read_excel_file(file_path: str) -> str:
    """Read an Excel (.xlsx/.xls) or CSV file and return its contents as text."""
    try:
        import pandas as pd
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".csv":
            df_dict = {"Sheet1": pd.read_csv(file_path)}
        elif ext in (".xlsx", ".xls"):
            df_dict = pd.read_excel(file_path, sheet_name=None)
        else:
            try:
                df_dict = {"Sheet1": pd.read_csv(file_path)}
            except Exception:
                return f"[read_excel_file error] Unsupported extension: {ext}"

        parts = []
        for sheet_name, df in df_dict.items():
            parts.append(f"=== Sheet: {sheet_name} ({df.shape[0]} rows x {df.shape[1]} cols) ===")
            parts.append(df.to_string(index=True, max_rows=300))
        return "\n".join(parts)
    except Exception as e:
        return f"[read_excel_file error] {e}"


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join(seg.text.strip() for seg in segments)
        return transcript.strip() or "[Empty transcript]"
    except ImportError:
        pass
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result.get("text", "").strip()
    except Exception as e:
        return f"[transcribe_audio error] {e}"


def describe_image(image_path: str) -> str:
    """Describe an image using HF Inference API (Qwen2.5-VL)."""
    import base64, io
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return "[describe_image error] HF_TOKEN not set"
    try:
        from PIL import Image
        from huggingface_hub import InferenceClient
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        client = InferenceClient(token=hf_token)
        response = client.chat_completion(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": (
                        "Describe this image in full detail. Include:\n"
                        "1. All visible text and numbers (OCR)\n"
                        "2. Objects, people, animals present\n"
                        "3. Any charts/tables — list all values\n"
                        "4. Colors, layout, spatial relationships\n"
                        "Be exhaustive and precise."
                    )}
                ]
            }],
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[describe_image error] {e}"


def get_youtube_transcript(url_or_id: str) -> str:
    """Fetch the transcript of a YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import urllib.parse
        vid = url_or_id.strip()
        if "youtube.com/watch" in vid:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(vid).query)
            vid = params.get("v", [vid])[0]
        elif "youtu.be/" in vid:
            vid = vid.split("youtu.be/")[-1].split("?")[0]
        entries = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US", "en-GB"])
        return " ".join(e["text"] for e in entries).strip()
    except Exception as e:
        return f"[get_youtube_transcript error] {e}"


# =============================================================================
# AGENT SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert AI agent solving GAIA benchmark tasks. GAIA uses EXACT string matching.

CRITICAL FORMATTING RULES:
1. Return ONLY the answer — no preamble, no explanation, no markdown
2. Numbers: use digits (42), no commas (1000 not 1,000), no units unless asked
3. Lists: comma-separated, no brackets: apple, banana, cherry
4. Strings: preserve exact spelling/capitalization from the source
5. Dates: match the format asked; default YYYY-MM-DD if unspecified
6. Names: give the full name exactly as it appears in the source

AVAILABLE TOOLS:
- search_web(query) → search the web
- fetch_webpage(url) → read a webpage in full
- download_file(task_id) → download attached file; returns text directly for CSV/TXT/JSON, or a local file path for audio/image/Excel
- read_spreadsheet(file_path) → read .xlsx/.xls/.csv files into a text table
- transcribe_audio_file(audio_path) → transcribe audio to text
- analyze_image(image_path) → describe image contents including all text/OCR and data
- youtube_transcript(url_or_id) → get YouTube video transcript

STRATEGY:
- If the question references an attached file/image/audio → call download_file(task_id) FIRST
- Route the returned path to the right tool (transcribe_audio_file, analyze_image, read_spreadsheet)
- For factual/web questions → search_web, then fetch_webpage for details
- For math/counting → compute precisely in Python
- Think step by step, then give ONLY the final answer"""


# =============================================================================
# AGENT
# =============================================================================

class BasicAgent:
    """
    GAIA Benchmark Agent using smolagents CodeAgent.
    Aligned with the official course template: __call__(question) -> str
    Task context (task_id, file_name) is injected into the question string
    by run_and_submit_all before calling the agent.
    """

    def __init__(self):
        from smolagents import CodeAgent, InferenceClientModel, tool

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("HF_TOKEN secret is not set. Add it in your HF Space settings.")

        model = InferenceClientModel(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            token=hf_token,
        )

        @tool
        def search_web(query: str) -> str:
            """Search the web for information. Returns titles, URLs, and snippets.
            Args:
                query: The search query string.
            """
            return web_search(query)

        @tool
        def fetch_webpage(url: str) -> str:
            """Fetch and return the text content of a webpage.
            Args:
                url: The URL to fetch.
            """
            return visit_webpage(url)

        @tool
        def download_file(task_id: str) -> str:
            """Download the file attached to a GAIA task. Returns text content directly for text/CSV/JSON files, or a local file path for audio/image/Excel files. Always call this first when a task mentions an attached file.
            Args:
                task_id: The GAIA task UUID.
            """
            return download_task_file(task_id)

        @tool
        def read_spreadsheet(file_path: str) -> str:
            """Read an Excel (.xlsx/.xls) or CSV file and return its data as formatted text.
            Args:
                file_path: Local path to the spreadsheet file.
            """
            return read_excel_file(file_path)

        @tool
        def transcribe_audio_file(audio_path: str) -> str:
            """Transcribe an audio file (mp3, wav, m4a, flac) to text.
            Args:
                audio_path: Local path to the audio file.
            """
            return transcribe_audio(audio_path)

        @tool
        def analyze_image(image_path: str) -> str:
            """Analyze an image and return a detailed description including all text, numbers, objects, and data visible in the image.
            Args:
                image_path: Local path to the image file.
            """
            return describe_image(image_path)

        @tool
        def youtube_transcript(url_or_id: str) -> str:
            """Fetch the full transcript of a YouTube video.
            Args:
                url_or_id: YouTube video URL or video ID.
            """
            return get_youtube_transcript(url_or_id)

        self.agent = CodeAgent(
            tools=[
                search_web,
                fetch_webpage,
                download_file,
                read_spreadsheet,
                transcribe_audio_file,
                analyze_image,
                youtube_transcript,
            ],
            model=model,
            max_steps=15,
            verbosity_level=1,
            additional_authorized_imports=[
                "os", "re", "json", "math", "datetime", "pathlib",
                "pandas", "numpy", "collections", "csv", "itertools",
                "string", "requests", "tempfile", "base64", "io",
                "PIL", "openpyxl",
            ],
        )

        print("BasicAgent initialized successfully.")

    def __call__(self, question: str) -> str:
        """Main interface — receives enriched question string, returns answer."""
        print(f"Agent received question (first 80 chars): {question[:80]}...")
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{question}"
            raw = self.agent.run(full_prompt, reset=True)
            return self._clean(str(raw))
        except Exception as e:
            print(f"[Agent error] {e}")
            traceback.print_exc()
            return ""

    @staticmethod
    def _clean(raw: str) -> str:
        s = raw.strip()
        if s.startswith("```") and s.endswith("```"):
            inner = s[3:-3].strip()
            first_line, _, rest = inner.partition("\n")
            s = rest.strip() if first_line.isalpha() else inner
        s = re.sub(
            r"^\s*(the\s+)?(final\s+)?(answer\s+(is|:)|result\s*:|value\s*:)\s*",
            "", s, flags=re.IGNORECASE,
        ).strip()
        for q in ('"', "'"):
            if s.startswith(q) and s.endswith(q) and len(s) > 1:
                s = s[1:-1].strip()
                break
        return s


# =============================================================================
# BENCHMARK RUNNER  (exact template structure)
# =============================================================================

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") or ""

        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue

        # Enrich the question with task context so the agent can use tools
        enriched_question = f"Task ID: {task_id}\n"
        if file_name:
            ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "file"
            type_hints = {
                "mp3": "audio", "wav": "audio", "m4a": "audio", "flac": "audio",
                "png": "image", "jpg": "image", "jpeg": "image", "gif": "image",
                "csv": "CSV spreadsheet", "xlsx": "Excel spreadsheet", "xls": "Excel spreadsheet",
                "txt": "text file", "json": "JSON file", "pdf": "PDF document",
            }
            ftype = type_hints.get(ext, "file")
            enriched_question += (
                f"Attached {ftype}: '{file_name}'\n"
                f"→ Call download_task_file(task_id='{task_id}') to access it BEFORE answering.\n"
            )
        enriched_question += f"\nQuestion: {question_text}"

        try:
            submitted_answer = agent(enriched_question)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer,
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}",
            })

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return final_status, pd.DataFrame(results_log)
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)


# =============================================================================
# GRADIO UI  (exact template structure)
# =============================================================================

with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
**Instructions:**

1. Make sure `HF_TOKEN` is set in your Space secrets (Settings → Variables and secrets).
2. Log in to your Hugging Face account using the button below.
3. Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

---
**Note:** This can take 20–30 minutes to run all 20 questions.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️ SPACE_HOST not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
    else:
        print("ℹ️ SPACE_ID not found (running locally?).")

    print("-" * (60 + len(" App Starting ")) + "\n")
    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
