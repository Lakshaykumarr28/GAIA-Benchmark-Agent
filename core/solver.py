import re
import traceback

from smolagents import CodeAgent, InferenceClientModel
from smolagents import DuckDuckGoSearchTool, VisitWebpageTool

from core.prompts import ANSWER_RULES
from mytools.download import DownloadTaskFileTool
from mytools.audio import AudioTranscriptionTool
from mytools.youtube import YouTubeTranscriptTool
from mytools.image import ImageDescriptionTool
from mytools.excel import ExcelReadTool


class GAIASolver:
    """
    Wraps a smolagents CodeAgent to solve GAIA benchmark tasks.

    Each task dict is expected to contain:
        task_id   : str — UUID used to fetch attached files from the API
        question  : str — The natural language question to answer
        file_name : str — Optional filename hint (may be empty string)
    """

    def __init__(self, model: InferenceClientModel):
        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
                YouTubeTranscriptTool(),
                AudioTranscriptionTool(),
                DownloadTaskFileTool(),
                ImageDescriptionTool(),
                ExcelReadTool(),
            ],
            model=model,
            max_steps=15,
            verbosity_level=1,
            additional_authorized_imports=[
                "pandas", "numpy", "re", "math", "datetime",
                "collections", "json", "csv", "itertools", "string",
                "openpyxl", "PIL", "pathlib", "os",
            ],
        )

    def solve(self, task: dict) -> str:
        """Solve one GAIA task and return a cleaned answer string."""
        task_id = task.get("task_id", "")
        question = task.get("question", "")
        file_name = task.get("file_name", "") or ""

        # Determine file type hint for the agent
        file_hint = ""
        if file_name:
            ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            type_hints = {
                "mp3": "audio", "wav": "audio", "m4a": "audio", "flac": "audio",
                "png": "image", "jpg": "image", "jpeg": "image", "gif": "image",
                "csv": "spreadsheet/text", "xlsx": "Excel spreadsheet", "xls": "Excel spreadsheet",
                "txt": "text", "json": "JSON data", "pdf": "PDF document",
            }
            ftype = type_hints.get(ext, "file")
            file_hint = (
                f"\nThis task has an attached {ftype}: '{file_name}'.\n"
                f"Call download_task_file(task_id='{task_id}') to access it BEFORE answering."
            )

        prompt = f"{ANSWER_RULES}\n---\nTask ID: {task_id}{file_hint}\n\nQuestion: {question}"

        try:
            raw = self.agent.run(prompt, reset=True)
            return self._clean(str(raw))
        except Exception as exc:
            print(f"[ERROR] Task {task_id}: {exc}")
            traceback.print_exc()
            return ""

    @staticmethod
    def _clean(raw: str) -> str:
        """
        Minimal post-processing for exact-match scoring.
        Conservative — avoids corrupting valid answers.
        """
        s = raw.strip()

        # Strip markdown code fences
        if s.startswith("```") and s.endswith("```"):
            inner = s[3:-3].strip()
            first_line, _, rest = inner.partition("\n")
            s = rest.strip() if first_line.isalpha() else inner

        # Strip common preamble patterns
        s = re.sub(
            r"^\s*(the\s+)?(final\s+)?(answer\s+(is|:)|result\s*:|value\s*:)\s*",
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()

        # Strip symmetric surrounding quotes
        for q in ('"', "'"):
            if s.startswith(q) and s.endswith(q) and len(s) > 1:
                s = s[1:-1].strip()
                break

        return s
