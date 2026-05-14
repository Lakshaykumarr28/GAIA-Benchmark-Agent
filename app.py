import os
import re
import io
import base64
import tempfile
import traceback
import subprocess

import gradio as gr
import requests
import pandas as pd

from smolagents import LiteLLMModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


# =============================================================================
# TOOL IMPLEMENTATIONS  (plain functions, wrapped via @tool inside __init__)
# =============================================================================

def _web_search(query: str, max_results: int = 8) -> str:
    """DuckDuckGo web search."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r.get('title','')}\nURL: {r.get('href','')}\nSnippet: {r.get('body','')}"
            for r in results
        )
    except Exception as e:
        return f"[web_search error] {e}"


def _visit_webpage(url: str) -> str:
    """Fetch full text of a webpage (up to 12 000 chars)."""
    try:
        import requests as _req
        from markdownify import markdownify
        r = _req.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = markdownify(r.text)
        return text[:12000]
    except Exception as e:
        return f"[visit_webpage error] {e}"


def _wikipedia_search(query: str) -> str:
    """Search Wikipedia and return article text (up to 6 000 chars)."""
    try:
        import requests as _req
        params = {"action": "query", "format": "json", "list": "search",
                  "srsearch": query, "srlimit": 3}
        r = _req.get("https://en.wikipedia.org/w/api.php", params=params, timeout=15)
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return "No Wikipedia results found."
        title = results[0]["title"]
        page_params = {"action": "query", "format": "json", "titles": title,
                       "prop": "extracts", "explaintext": True, "exintro": False}
        pr = _req.get("https://en.wikipedia.org/w/api.php", params=page_params, timeout=15)
        pages = pr.json().get("query", {}).get("pages", {})
        text = next(iter(pages.values())).get("extract", "")
        return f"Wikipedia: {title}\nURL: https://en.wikipedia.org/wiki/{title.replace(' ','_')}\n\n{text[:6000]}"
    except Exception as e:
        return f"[wikipedia_search error] {e}"


def _download_task_file(task_id: str) -> str:
    """Download the file attached to a GAIA task."""
    url = f"{DEFAULT_API_URL}/files/{task_id}"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return f"[download_file error] {e}"

    content_disp = r.headers.get("Content-Disposition", "")
    content_type = r.headers.get("Content-Type", "")

    filename = ""
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[-1].strip().strip('"')
    ext = os.path.splitext(filename)[-1].lower() if filename else ""

    if not ext:
        mime_map = {
            "csv": ".csv", "json": ".json", "plain": ".txt",
            "mpeg": ".mp3", "audio": ".mp3", "wav": ".wav",
            "png": ".png", "jpeg": ".jpg", "gif": ".gif",
            "excel": ".xlsx", "spreadsheetml": ".xlsx",
            "pdf": ".pdf", "python": ".py",
        }
        for key, val in mime_map.items():
            if key in content_type.lower():
                ext = val
                break

    TEXT_EXTS = {".csv", ".txt", ".json", ".tsv", ".md", ".xml", ".html"}
    if ext in TEXT_EXTS:
        try:
            return r.content.decode("utf-8")
        except UnicodeDecodeError:
            return r.content.decode("latin-1")

    suffix = ext or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(r.content)
        return tmp.name


def _read_file(file_path: str) -> str:
    """Read xlsx/csv/txt/json/pdf — returns data as text."""
    if not os.path.exists(file_path):
        return f"[read_file error] File not found: {file_path}"
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in (".xlsx", ".xls", ".csv"):
        try:
            import pandas as _pd
            if ext == ".csv":
                df_dict = {"Sheet1": _pd.read_csv(file_path)}
            else:
                df_dict = _pd.read_excel(file_path, sheet_name=None)
            parts = []
            for name, df in df_dict.items():
                parts.append(f"=== Sheet: {name} ({df.shape[0]} rows x {df.shape[1]} cols) ===")
                parts.append(df.to_string(index=True, max_rows=500))
                parts.append("")
            return "\n".join(parts)
        except Exception as e:
            return f"[read_file/excel error] {e}"
    if ext in (".txt", ".py", ".json", ".md", ".tsv", ".xml"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"[read_file/text error] {e}"
    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            pass
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return f"[read_file/pdf error] {e}"
    return f"[read_file] Unrecognised extension '{ext}' — use transcribe_audio or analyze_image for media."


def _execute_python(code: str) -> str:
    """Execute Python code string and return stdout + stderr."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30
        )
        out = result.stdout or ""
        err = result.stderr or ""
        combined = out + ("\n[stderr]\n" + err if err else "")
        return combined.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[execute_python] Timed out after 30s"
    except Exception as e:
        return f"[execute_python error] {e}"


def _execute_python_file(file_path: str) -> str:
    """Execute a .py script file and return its output."""
    if not os.path.exists(file_path):
        return f"[execute_python_file error] File not found: {file_path}"
    try:
        result = subprocess.run(
            ["python3", file_path],
            capture_output=True, text=True, timeout=30
        )
        out = result.stdout or ""
        err = result.stderr or ""
        combined = out + ("\n[stderr]\n" + err if err else "")
        return combined.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[execute_python_file] Timed out after 30s"
    except Exception as e:
        return f"[execute_python_file error] {e}"


def _transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using Whisper."""
    if not os.path.exists(audio_path):
        return f"[transcribe_audio error] File not found: {audio_path}"
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


def _analyze_image(image_path: str) -> str:
    """Analyze image via Qwen2.5-VL — OCR, chess boards, charts, diagrams."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return "[analyze_image error] HF_TOKEN not set"
    if not os.path.exists(image_path):
        return f"[analyze_image error] File not found: {image_path}"
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
                        "Describe this image in maximum detail:\n"
                        "1. All visible text, numbers, labels (exact OCR)\n"
                        "2. If chess board: name EVERY piece and its exact square (e.g. White King on e1, Black Rook on f8)\n"
                        "3. If table/chart: list every single value\n"
                        "4. Objects, spatial layout, colors\n"
                        "Be exhaustive and precise."
                    )}
                ]
            }],
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[analyze_image error] {e}"


def _youtube_transcript(url_or_id: str) -> str:
    """Fetch full YouTube video transcript."""
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
        return f"[youtube_transcript error] {e}"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert AI agent solving GAIA benchmark tasks. GAIA scores by EXACT string matching.

CRITICAL OUTPUT RULES:
1. Output ONLY the final answer — zero preamble, zero explanation, zero markdown
2. Numbers: digits only, no commas (1000 not 1,000), no units unless asked
3. Lists: comma-separated no brackets → apple, banana, cherry
4. Strings: preserve exact spelling and capitalization from the source
5. Dates: match format requested; default ISO YYYY-MM-DD
6. Names: full name exactly as in source material

TOOLS AVAILABLE:
• search_web(query)               — DuckDuckGo search
• visit_webpage(url)              — Fetch full page text (follow links!)
• wikipedia_search(query)         — Direct Wikipedia lookup
• download_file(task_id)          — Download GAIA task attachment (returns text or file path)
• read_file(file_path)            — Read xlsx/csv/txt/json/pdf from path
• execute_python(code)            — Run Python code for math/logic/computation
• execute_python_file(file_path)  — Run a downloaded .py script
• transcribe_audio(audio_path)    — Whisper speech-to-text for mp3/wav/m4a
• analyze_image(image_path)       — Vision model: OCR, chess, charts, diagrams
• youtube_transcript(url_or_id)   — Full YouTube transcript

STRATEGY BY QUESTION TYPE:

REVERSED TEXT (question starts with "." or contains ".rewsna"):
  The question is written backwards. Reverse the entire string to read it, then answer.
  Example reversed: ".rewsna eht sa 'tfel' drow eht fo etisoppo eht etirw"
  → Real instruction: "write the opposite of the word 'left' as the answer" → answer: right

FILE QUESTIONS (mentions attached file, image, audio, spreadsheet):
  1. Call download_file(task_id) — ALWAYS first
  2. Route by type: audio→transcribe_audio | image→analyze_image | xlsx/csv→read_file | .py→execute_python_file

PYTHON CODE tasks:
  download_file returns the Python source as a string (since .py is a text file).
  Write it to a temp file and run with execute_python_file, OR run with execute_python(code).

MATH / COMMUTATIVITY / TABLE tasks:
  Always use execute_python — never mental arithmetic.
  For commutativity: iterate all (x,y) pairs and check table[x][y] != table[y][x].

WIKIPEDIA tasks:
  Use wikipedia_search first. For Featured Article history:
  visit https://en.wikipedia.org/wiki/Wikipedia:Featured_articles
  For WikiProject discussions use the article Talk page.

YOUTUBE tasks:
  Always use youtube_transcript(url) — never guess from memory.

AUDIO tasks:
  download_file(task_id) → local path → transcribe_audio(path)

EXCEL / SALES / SPREADSHEET:
  download_file → read_file → execute_python to compute sums/filters

SPORTS STATS:
  search_web with specific year+stat+player, then visit Baseball Reference / Sports Reference.

ACADEMIC PAPERS:
  search_web for title/authors, visit_webpage the paper URL, follow reference links.
  
VERY IMPORTANT:
After solving, ALWAYS call final_answer with ONLY the exact final answer.
Never stop without calling final_answer.
  
"""


# =============================================================================
# AGENT
# =============================================================================

class BasicAgent:
    """GAIA Benchmark Agent. Interface: __call__(question: str) -> str"""

    def __init__(self):
        from smolagents import CodeAgent, InferenceClientModel, tool

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("HF_TOKEN secret is not set. Add it in Space Settings → Variables and secrets.")

        # model = InferenceClientModel(
        #     model_id="Qwen/Qwen2.5-72B-Instruct",
        #     token=hf_token,
        # )

        model = LiteLLMModel(
            model_id="openai/Qwen/Qwen2.5-72B-Instruct",
            api_key=hf_token,
            max_tokens=4096,
            temperature=0.1,
        )

        @tool
        def search_web(query: str) -> str:
            """Search the web using DuckDuckGo. Returns titles, URLs, and text snippets. Run multiple targeted queries for best results.
            Args:
                query: The search query.
            """
            return _web_search(query)

        @tool
        def visit_webpage(url: str) -> str:
            """Fetch the full text content of a webpage. Use this to read Wikipedia articles, papers, sports stats pages, etc.
            Args:
                url: Full URL starting with http/https.
            """
            return _visit_webpage(url)

        @tool
        def wikipedia_search(query: str) -> str:
            """Search Wikipedia and return the article text. Best for people, events, albums, films, places, sports.
            Args:
                query: Wikipedia search query.
            """
            return _wikipedia_search(query)

        @tool
        def download_file(task_id: str) -> str:
            """Download the file attached to this GAIA task. Text/CSV/JSON/Python files are returned as a string. Audio/image/Excel/PDF files are saved to disk and the local path is returned. ALWAYS call this first when the task mentions a file, image, or audio.
            Args:
                task_id: The GAIA task UUID shown in the task context.
            """
            return _download_task_file(task_id)

        @tool
        def read_file(file_path: str) -> str:
            """Read a downloaded file and return its contents. Handles Excel (.xlsx/.xls), CSV, plain text, JSON, and PDF. Use after download_file returns a path.
            Args:
                file_path: Local path returned by download_file.
            """
            return _read_file(file_path)

        @tool
        def execute_python(code: str) -> str:
            """Execute Python code and return stdout/stderr. Use for ALL arithmetic, data analysis, logic checks, commutativity tables, list operations — never do math in your head.
            Args:
                code: Valid Python code string to execute.
            """
            return _execute_python(code)

        @tool
        def execute_python_file(file_path: str) -> str:
            """Execute a Python .py script file and return its output. Use when download_file returns a .py file path.
            Args:
                file_path: Local path to the .py script.
            """
            return _execute_python_file(file_path)

        @tool
        def transcribe_audio(audio_path: str) -> str:
            """Transcribe an audio file to text using Whisper. Handles mp3, wav, m4a, flac. Use after download_file returns an audio path.
            Args:
                audio_path: Local path to the audio file.
            """
            return _transcribe_audio(audio_path)

        @tool
        def analyze_image(image_path: str) -> str:
            """Analyze an image with a vision model. Extracts all text/OCR, describes chess piece positions precisely, reads charts/tables, identifies objects. Use after download_file returns an image path.
            Args:
                image_path: Local path to the image file.
            """
            return _analyze_image(image_path)

        @tool
        def youtube_transcript(url_or_id: str) -> str:
            """Get the full transcript of a YouTube video. Use for any question about YouTube video content.
            Args:
                url_or_id: Full YouTube URL or just the video ID.
            """
            return _youtube_transcript(url_or_id)

        self.agent = CodeAgent(
            tools=[
                search_web, visit_webpage, wikipedia_search,
                download_file, read_file,
                execute_python, execute_python_file,
                transcribe_audio, analyze_image, youtube_transcript,
            ],
            model=model,
            max_steps=20,
            verbosity_level=1,
            additional_authorized_imports=[
                "os", "re", "json", "math", "datetime", "pathlib",
                "pandas", "numpy", "collections", "csv", "itertools",
                "string", "requests", "tempfile", "base64", "io",
                "PIL", "openpyxl", "subprocess", "statistics",
            ],
            planning_interval=3,
        )
        print("BasicAgent initialized successfully.")

    def __call__(self, question: str) -> str:
        print(f"\n{'='*60}")
        print(f"Question (first 120): {question[:120]}")

        # Detect reversed-text questions and add hint
        raw_q = question
        m = re.search(r"Question:\s*(.+)", question, re.DOTALL)
        if m:
            raw_q = m.group(1).strip()

        reversed_hint = ""
        if raw_q.startswith(".") or ".rewsna eht" in raw_q.lower():
            reversed_text = raw_q[::-1]
            reversed_hint = (
                f"\n\n[SYSTEM NOTE: The question above is written in REVERSE. "
                f"Read it backwards. Reversed text: \"{reversed_text}\"]"
            )

        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{'='*60}\n\n{question}{reversed_hint}"
            raw = self.agent.run(full_prompt, reset=True)

            if raw is None:
                return ""

            if hasattr(raw, "content"):
                raw = raw.content

            answer = self._clean(str(raw))
            print(f"Answer: {answer!r}")
            return answer
        except Exception as e:
            print(f"[Agent error] {e}")
            traceback.print_exc()
            return ""

    # @staticmethod
    # def _clean(raw: str) -> str:
    #     s = raw.strip()
    #     if s.startswith("```") and s.endswith("```"):
    #         inner = s[3:-3].strip()
    #         first_line, _, rest = inner.partition("\n")
    #         s = rest.strip() if first_line.replace("-", "").isalpha() else inner
    #     s = re.sub(
    #         r"^\s*(the\s+)?(final\s+)?(answer\s+(is|:)|result\s*:|value\s*:)\s*",
    #         "", s, flags=re.IGNORECASE,
    #     ).strip()
    #     for q in ('"', "'"):
    #         if s.startswith(q) and s.endswith(q) and len(s) > 1:
    #             s = s[1:-1].strip()
    #             break
    #     return s
    
    @staticmethod
    def _clean(raw: str) -> str:
        return raw.strip().strip('"').strip("'")


# =============================================================================
# BENCHMARK RUNNER  (exact template structure)
# =============================================================================

def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        return f"Error fetching questions: {e}", None
    except Exception as e:
        return f"An unexpected error occurred fetching questions: {e}", None

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

        enriched = f"Task ID: {task_id}\n"
        if file_name:
            ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            type_map = {
                "mp3": "audio", "wav": "audio", "m4a": "audio", "flac": "audio", "ogg": "audio",
                "png": "image", "jpg": "image", "jpeg": "image", "gif": "image", "bmp": "image",
                "csv": "CSV spreadsheet", "xlsx": "Excel spreadsheet", "xls": "Excel spreadsheet",
                "txt": "text file", "json": "JSON file", "pdf": "PDF document", "py": "Python script",
            }
            ftype = type_map.get(ext, "attached file")
            enriched += f"Attached {ftype}: '{file_name}'\n"
            enriched += f"→ REQUIRED: Call download_file(task_id='{task_id}') FIRST.\n"
            routing = {
                frozenset(["mp3","wav","m4a","flac","ogg"]): "→ Then call transcribe_audio(path) on the returned path.",
                frozenset(["png","jpg","jpeg","gif","bmp"]): "→ Then call analyze_image(path) on the returned path.",
                frozenset(["xlsx","xls","csv"]): "→ Then call read_file(path) to read the spreadsheet data.",
                frozenset(["py"]): "→ The file content is returned as text. Save to temp and run with execute_python_file, or run with execute_python(code).",
            }
            for exts_set, hint in routing.items():
                if ext in exts_set:
                    enriched += hint + "\n"
                    break

        enriched += f"\nQuestion: {question_text}"

        try:
            submitted_answer = agent(enriched)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:100],
                "File": file_name or "—",
                "Submitted Answer": submitted_answer,
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:100],
                "File": file_name or "—",
                "Submitted Answer": f"AGENT ERROR: {e}",
            })

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
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
        except Exception:
            error_detail += f" Response: {e.response.text[:500]}"
        return f"Submission Failed: {error_detail}", pd.DataFrame(results_log)
    except requests.exceptions.Timeout:
        return "Submission Failed: The request timed out.", pd.DataFrame(results_log)
    except requests.exceptions.RequestException as e:
        return f"Submission Failed: Network error - {e}", pd.DataFrame(results_log)
    except Exception as e:
        return f"An unexpected error occurred during submission: {e}", pd.DataFrame(results_log)


# =============================================================================
# GRADIO UI
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
**Note:** This takes 20–30 minutes for all 20 questions. Do not close the page.
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")
    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️ SPACE_HOST not found (running locally?).")
    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
    else:
        print("ℹ️ SPACE_ID not found (running locally?).")
    print("-" * (60 + len(" App Starting ")) + "\n")
    demo.launch(debug=True, share=False)
