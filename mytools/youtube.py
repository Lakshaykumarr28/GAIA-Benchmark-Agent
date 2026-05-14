from smolagents import Tool


class YouTubeTranscriptTool(Tool):
    """
    Fetches the transcript of a YouTube video given its URL or video ID.
    """

    name = "get_youtube_transcript"
    description = (
        "Fetches the full transcript of a YouTube video. "
        "Pass the full YouTube URL (e.g. https://www.youtube.com/watch?v=...) "
        "or just the video ID. Returns the transcript as plain text."
    )
    inputs = {
        "url_or_id": {
            "type": "string",
            "description": "YouTube video URL or video ID string.",
        }
    }
    output_type = "string"

    def forward(self, url_or_id: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            # Extract video ID from URL if needed
            video_id = url_or_id.strip()
            if "youtube.com/watch" in video_id:
                import urllib.parse
                parsed = urllib.parse.urlparse(video_id)
                params = urllib.parse.parse_qs(parsed.query)
                video_id = params.get("v", [video_id])[0]
            elif "youtu.be/" in video_id:
                video_id = video_id.split("youtu.be/")[-1].split("?")[0]

            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en", "en-US", "en-GB"]
            )
            text = " ".join(entry["text"] for entry in transcript_list)
            return text.strip()

        except Exception as exc:
            return f"[YouTubeTranscriptTool ERROR] {exc}"
