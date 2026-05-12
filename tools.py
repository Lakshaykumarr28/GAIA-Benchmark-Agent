
from smolagents import tool

import os
import io
import requests
import pandas as pd
import duckdb

from PIL import Image
from bs4 import BeautifulSoup

from transformers import pipeline

from youtube_transcript_api import YouTubeTranscriptApi

from faster_whisper import WhisperModel


# =========================================================
# GLOBAL MODEL LOADERS
# =========================================================

# ---------------------------
# IMAGE CAPTIONING MODEL
# ---------------------------

image_captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

# ---------------------------
# SPEECH TO TEXT MODEL
# ---------------------------

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

# =========================================================
# IMAGE DESCRIPTION TOOL
# =========================================================

@tool
def image_describe_tool(image_path: str) -> str:
    """
    Describes an image using BLIP image captioning model.

    Args:
        image_path: Path to image file.

    Returns:
        Generated image description.
    """

    try:
        image = Image.open(image_path)

        result = image_captioner(image)

        caption = result[0]["generated_text"]

        return f"Image Description: {caption}"

    except Exception as e:
        return f"Error processing image: {str(e)}"


# =========================================================
# WEB SEARCH TOOL
# =========================================================

@tool
def web_search_tool(query: str) -> str:
    """
    Performs a basic web search using DuckDuckGo HTML search.

    Args:
        query: Search query.

    Returns:
        Top search results summary.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        url = f"https://html.duckduckgo.com/html/?q={query}"

        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")

        results = soup.find_all("a", class_="result__a")

        output = []

        for idx, result in enumerate(results[:5]):

            title = result.get_text()
            link = result.get("href")

            output.append(
                f"{idx+1}. {title}\nURL: {link}\n"
            )

        if not output:
            return "No search results found."

        return "\n".join(output)

    except Exception as e:
        return f"Search Error: {str(e)}"


# =========================================================
# CSV READER TOOL
# =========================================================

@tool
def read_csv_tool(file_path: str) -> str:
    """
    Reads CSV file and returns basic information.

    Args:
        file_path: Path to CSV file.

    Returns:
        Summary of CSV file.
    """

    try:
        df = pd.read_csv(file_path)

        info = f"""
CSV FILE SUMMARY

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names:
{list(df.columns)}

First 5 Rows:
{df.head().to_string()}
"""

        return info

    except Exception as e:
        return f"CSV Read Error: {str(e)}"


# =========================================================
# EXCEL READER TOOL
# =========================================================

@tool
def read_excel_tool(file_path: str) -> str:
    """
    Reads Excel file and returns sheet information.

    Args:
        file_path: Path to Excel file.

    Returns:
        Excel summary.
    """

    try:
        excel_file = pd.ExcelFile(file_path)

        sheets = excel_file.sheet_names

        output = [f"Available Sheets: {sheets}\n"]

        for sheet in sheets:

            df = pd.read_excel(file_path, sheet_name=sheet)

            output.append(
                f"""
Sheet: {sheet}

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Columns:
{list(df.columns)}

Preview:
{df.head().to_string()}
"""
            )

        return "\n".join(output)

    except Exception as e:
        return f"Excel Read Error: {str(e)}"


# =========================================================
# PARQUET READER TOOL
# =========================================================

@tool
def read_parquet_tool(file_path: str) -> str:
    """
    Reads parquet file and returns summary.

    Args:
        file_path: Path to parquet file.

    Returns:
        Parquet summary.
    """

    try:
        df = pd.read_parquet(file_path)

        return f"""
PARQUET FILE SUMMARY

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Columns:
{list(df.columns)}

Preview:
{df.head().to_string()}
"""

    except Exception as e:
        return f"Parquet Read Error: {str(e)}"


# =========================================================
# DATAFRAME QUERY TOOL
# =========================================================

@tool
def dataframe_query_tool(file_path: str, query: str) -> str:
    """
    Query CSV/Excel/Parquet files using SQL.

    Supported:
    - CSV
    - Excel
    - Parquet

    Args:
        file_path: Data file path.
        query: SQL query.

    Example:
        SELECT * FROM data LIMIT 5

    Returns:
        Query result.
    """

    try:

        extension = file_path.split(".")[-1]

        if extension == "csv":
            df = pd.read_csv(file_path)

        elif extension in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)

        elif extension == "parquet":
            df = pd.read_parquet(file_path)

        else:
            return "Unsupported file format."

        con = duckdb.connect()

        con.register("data", df)

        result = con.execute(query).fetchdf()

        return result.to_string()

    except Exception as e:
        return f"Query Error: {str(e)}"


# =========================================================
# DATAFRAME COLUMN ANALYSIS TOOL
# =========================================================

@tool
def dataframe_column_stats_tool(file_path: str, column_name: str) -> str:
    """
    Generates statistics for a column.

    Args:
        file_path: Path to data file.
        column_name: Column name.

    Returns:
        Statistical summary.
    """

    try:

        extension = file_path.split(".")[-1]

        if extension == "csv":
            df = pd.read_csv(file_path)

        elif extension in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)

        elif extension == "parquet":
            df = pd.read_parquet(file_path)

        else:
            return "Unsupported format."

        if column_name not in df.columns:
            return "Column not found."

        stats = df[column_name].describe()

        return stats.to_string()

    except Exception as e:
        return f"Stats Error: {str(e)}"


# =========================================================
# SPEECH TO TEXT TOOL
# =========================================================

@tool
def speech_to_text_tool(audio_path: str) -> str:
    """
    Converts speech audio to text using Whisper.

    Args:
        audio_path: Path to audio file.

    Returns:
        Transcribed text.
    """

    try:

        segments, info = whisper_model.transcribe(audio_path)

        transcription = ""

        for segment in segments:
            transcription += segment.text + " "

        return transcription.strip()

    except Exception as e:
        return f"Transcription Error: {str(e)}"


# =========================================================
# GENERIC FILE READER TOOL
# =========================================================

@tool
def read_file_tool(file_path: str) -> str:
    """
    Reads generic text files.

    Supported:
    - txt
    - md
    - py
    - json
    - csv
    - log

    Args:
        file_path: File path.

    Returns:
        File contents.
    """

    try:

        with open(file_path, "r", encoding="utf-8") as f:

            content = f.read()

        return content[:15000]

    except Exception as e:
        return f"File Read Error: {str(e)}"


# =========================================================
# YOUTUBE TRANSCRIPT TOOL
# =========================================================

@tool
def youtube_transcript_tool(video_id: str) -> str:
    """
    Fetches YouTube video transcript.

    Args:
        video_id: YouTube video ID.

    Example:
        dQw4w9WgXcQ

    Returns:
        Video transcript.
    """

    try:

        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        full_text = " ".join(
            [item["text"] for item in transcript]
        )

        return full_text

    except Exception as e:
        return f"Transcript Error: {str(e)}"


# =========================================================
# OPTIONAL: YOUTUBE URL PARSER TOOL
# =========================================================

@tool
def extract_youtube_video_id_tool(url: str) -> str:
    """
    Extracts YouTube video ID from URL.

    Args:
        url: Full YouTube URL.

    Returns:
        Video ID.
    """

    try:

        if "v=" in url:
            return url.split("v=")[1].split("&")[0]

        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1]

        else:
            return "Invalid YouTube URL"

    except Exception as e:
        return f"URL Parse Error: {str(e)}"
