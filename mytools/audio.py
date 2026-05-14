import os
import tempfile

import requests
from smolagents import Tool


class AudioTranscriptionTool(Tool):
    """
    Transcribes an audio file to text using faster-whisper (CPU-friendly).
    Accepts a local file path returned by DownloadTaskFileTool.
    """

    name = "transcribe_audio"
    description = (
        "Transcribes an audio file (mp3, wav, m4a, flac, ogg) to text. "
        "Pass the local file path returned by download_task_file. "
        "Returns the full transcript as a string."
    )
    inputs = {
        "audio_path": {
            "type": "string",
            "description": "Local path to the audio file to transcribe.",
        }
    }
    output_type = "string"

    def forward(self, audio_path: str) -> str:
        # Strip any prefix message from DownloadTaskFileTool
        if "Audio file saved to:" in audio_path:
            audio_path = audio_path.split("Audio file saved to:")[-1].split("\n")[0].strip()

        if not os.path.exists(audio_path):
            return f"[AudioTranscriptionTool ERROR] File not found: {audio_path}"

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(audio_path, beam_size=5)
            transcript = " ".join(seg.text.strip() for seg in segments)
            return transcript.strip() if transcript.strip() else "[Empty transcript]"
        except ImportError:
            pass

        # Fallback: try openai-whisper
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return result.get("text", "").strip()
        except Exception as exc:
            return f"[AudioTranscriptionTool ERROR] {exc}"
