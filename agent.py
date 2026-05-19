import os
from urllib import response
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from docx import Document
from dotenv import load_dotenv 
# OCR / Vision
from PIL import Image
import pytesseract
import wikipedia
from requests.exceptions import RequestException
from json.decoder import JSONDecodeError
import re
import whisper

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

tavily = TavilySearch(
    # tavily_api_key=TAVILY_API_KEY,
    max_results=1,
    topic="general",
)

whisper_model = whisper.load_model("base")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    ONLY for valid numeric mathematical calculations.

    Examples:
    - 17 * 42
    - (25 + 3) / 7

    DO NOT use for:
    - words
    - logic
    - reasoning
    - readability
    - text operations
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def wiki_search(query: str) -> str:
    """
    Search Wikipedia safely and return summarized results.
    """

    try:
        wikipedia.set_lang("en")

        results = wikipedia.search(query, results=2)

        if not results:
            return "No Wikipedia results found."

        docs = []

        for title in results:

            try:
                page = wikipedia.page(title, auto_suggest=False)

                docs.append(
                    f"Title: {page.title}\n"
                    f"Summary: {page.summary[:500]}"
                )

            except Exception:
                continue

        if not docs:
            return "No readable Wikipedia pages found."

        return "\n\n---\n\n".join(docs)

    except JSONDecodeError:
        return "Wikipedia API returned invalid response."

    except RequestException as e:
        return f"Wikipedia request failed: {str(e)}"

    except Exception as e:
        return f"Wikipedia search error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    Search the web ONLY when external factual information is required.

    USE THIS TOOL FOR:
    - current events
    - recent facts
    - real people
    - locations
    - companies
    - websites
    - historical facts
    - verifying uncertain information

    DO NOT USE THIS TOOL FOR:
    - math
    - logic
    - reversing text
    - summarization
    - rewriting
    - simple reasoning
    - extracting answers already present in the question
    - general common knowledge

    Input:
        query: concise factual search query

    Returns:
        summarized web search results
    """

    try:

        # prevent garbage searches
        query = query.strip()

        if len(query) < 3:
            return "Invalid search query."

        # prevent obvious unnecessary searches
        blocked_patterns = [
            "reverse text",
            "reverse string",
            "calculator",
            "math",
            "spell",
            "translate",
            "rewrite",
        ]

        lower_query = query.lower()

        if any(p in lower_query for p in blocked_patterns):
            return "Web search unnecessary for this query."

        response = tavily.invoke(query)

        # Tavily returns dict
        results = response.get("results", [])

        if not results:
            return "No web results found."

        formatted = []

        for doc in results[:3]:

            title = doc.get("title", "Unknown")
            url = doc.get("url", "")
            content = doc.get("content", "")[:500]

            formatted.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Web search error: {str(e)}"

@tool
def read_pdf(path: str) -> str:
    """
    Read text from a LOCAL PDF file only.

    USE ONLY FOR:
    - extracting PDF contents
    - reading uploaded PDF documents

    DO NOT USE FOR:
    - web search
    - reasoning
    - math
    - summarization without a PDF path

    Input:
        path: local PDF file path ending with .pdf
    """

    try:

        if not path.lower().endswith(".pdf"):
            return "Error: file must be a PDF."

        loader = PyPDFLoader(path)
        pages = loader.load()

        if not pages:
            return "Error: empty PDF."

        return "\n".join(
            p.page_content for p in pages[:5]
        )[:1000]

    except Exception as e:
        return f"PDF read error: {str(e)}"


@tool
def read_excel(path: str) -> str:
    """
    Read a LOCAL Excel file only.

    USE ONLY FOR:
    - .xlsx
    - .xls files

    DO NOT USE FOR:
    - calculations without a file
    - CSV files
    - web tasks

    Input:
        local Excel file path
    """

    try:

        if not path.lower().endswith((".xlsx", ".xls")):
            return "Error: file must be Excel format."

        df = pd.read_excel(path)

        if df.empty:
            return "Error: empty Excel file."

        return df.head(10).to_string()[:1000]

    except Exception as e:
        return f"Excel read error: {str(e)}"


@tool
def read_csv(path: str) -> str:
    """
    Read a LOCAL CSV file only.

    USE ONLY FOR:
    - CSV data inspection
    - reading tabular CSV data

    DO NOT USE FOR:
    - Excel files
    - web search
    - calculations without CSV input

    Input:
        local CSV file path
    """

    try:

        if not path.lower().endswith(".csv"):
            return "Error: file must be CSV format."

        df = pd.read_csv(path)

        if df.empty:
            return "Error: empty CSV file."

        return df.head(10).to_string()[:1000]

    except Exception as e:
        return f"CSV read error: {str(e)}"


@tool
def read_word(path: str) -> str:
    """
    Read a LOCAL Word document only.

    USE ONLY FOR:
    - .docx document reading

    DO NOT USE FOR:
    - PDFs
    - OCR
    - web tasks

    Input:
        local .docx file path
    """

    try:

        if not path.lower().endswith(".docx"):
            return "Error: file must be .docx format."

        doc = Document(path)

        text = "\n".join(
            p.text for p in doc.paragraphs
        ).strip()

        if not text:
            return "Error: empty Word document."

        return text[:1000]

    except Exception as e:
        return f"Word read error: {str(e)}"


@tool
def ocr_image(path: str) -> str:
    """
    Extract text from a LOCAL image file using OCR.

    USE ONLY FOR:
    - images containing text
    - OCR extraction

    DO NOT USE FOR:
    - PDFs
    - reasoning
    - image generation
    - web search

    Supported:
    .png .jpg .jpeg

    Input:
        local image file path
    """

    try:

        valid_ext = (".png", ".jpg", ".jpeg")

        if not path.lower().endswith(valid_ext):
            return "Error: unsupported image format."

        image = Image.open(path)

        text = pytesseract.image_to_string(image).strip()

        if not text:
            return "No text detected in image."

        return text[:1000]

    except Exception as e:
        return f"OCR error: {str(e)}"
    


@tool
def speech_to_text(path: str) -> str:
    """
    Convert LOCAL audio speech into text.

    USE ONLY FOR:
    - transcribing audio files
    - speech recognition
    - extracting spoken words

    DO NOT USE FOR:
    - normal text files
    - OCR
    - reasoning
    - web search

    Supported formats:
    .mp3 .wav .m4a .mp4

    Input:
        local audio/video file path
    """

    try:

        valid_ext = (
            ".mp3",
            ".wav",
            ".m4a",
            ".mp4"
        )

        if not path.lower().endswith(valid_ext):
            return "Error: unsupported audio format."

        if not os.path.exists(path):
            return "Error: audio file not found."

        result = whisper_model.transcribe(path)

        text = result.get("text", "").strip()

        if not text:
            return "No speech detected."

        return text[:1000]

    except Exception as e:
        return f"Speech-to-text error: {str(e)}"


# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(system_prompt)

# tools = [
#     calculator,
#     wiki_search,
#     web_search,
#     read_pdf,
#     read_excel,
#     read_csv,
#     read_word,
#     ocr_image,
#     speech_to_text,
# ]


ALL_TOOLS = {
    "calculator": calculator,
    "wiki_search": wiki_search,
    "web_search": web_search,
    "read_pdf": read_pdf,
    "read_excel": read_excel,
    "read_csv": read_csv,
    "read_word": read_word,
    "ocr_image": ocr_image,
    "speech_to_text": speech_to_text,
}


def get_tools(question: str):

    q = question.lower()

    selected_tools = []

    # always useful
    selected_tools.append(ALL_TOOLS["calculator"])

    # web / factual questions
    web_keywords = [
        "who",
        "when",
        "where",
        "latest",
        "news",
        "movie",
        "president",
        "capital",
        "highest grossing",
        "company",
        "history",
    ]

    if any(k in q for k in web_keywords):
        selected_tools.append(ALL_TOOLS["web_search"])
        selected_tools.append(ALL_TOOLS["wiki_search"])

    # pdf
    if ".pdf" in q:
        selected_tools.append(ALL_TOOLS["read_pdf"])

    # excel
    if ".xlsx" in q or ".xls" in q:
        selected_tools.append(ALL_TOOLS["read_excel"])

    # csv
    if ".csv" in q:
        selected_tools.append(ALL_TOOLS["read_csv"])

    # word
    if ".docx" in q:
        selected_tools.append(ALL_TOOLS["read_word"])

    # image / OCR
    image_exts = [".png", ".jpg", ".jpeg"]

    if any(ext in q for ext in image_exts):
        selected_tools.append(ALL_TOOLS["ocr_image"])

    # audio
    audio_exts = [".mp3", ".wav", ".m4a", ".mp4"]

    if any(ext in q for ext in audio_exts):
        selected_tools.append(ALL_TOOLS["speech_to_text"])

    return selected_tools


# Build graph function
def build_graph(question: str):

    tools = get_tools(question)

    llm = ChatGroq(
        model="gemma2-9b-it",
        temperature=0,
        max_tokens=128,
        groq_api_key=GROQ_API_KEY
    )

    chat_with_tools = llm.bind_tools(
        tools,
        parallel_tool_calls=False
    )

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state: AgentState):

        response = chat_with_tools.invoke(state["messages"])

        return {
            "messages": [response]
        }

    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)

    builder.add_node(
        "tools",
        ToolNode(
            tools,
            handle_tool_errors=True
        )
    )

    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    builder.add_edge("tools", "assistant")

    return builder.compile()



def extract_final_answer(messages):

    for msg in reversed(messages):

        if getattr(msg, "type", "") != "ai":
            continue

        content = str(getattr(msg, "content", "")).strip()

        if not content:
            continue

        # PRIMARY: FINAL ANSWER format
        match = re.search(
            r"FINAL ANSWER:\s*(.+)",
            content,
            re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(1).strip().splitlines()[0]

        # FALLBACK:
        # short direct answer from model
        if len(content.split()) <= 10:
            return content.strip()

    return "unknown"


def is_reversed_question(text: str, threshold: float = 0.6):
    """
    Detect whether a sentence is likely reversed.

    Returns:
        (is_reversed: bool, corrected_text: str)
    """

    # common English words
    common_words = {
        "the", "is", "are", "if", "you", "can",
        "what", "where", "when", "why", "how",
        "question", "answer", "give", "read",
        "this", "that", "and", "or", "to"
    }

    # tokenize original
    words = re.findall(r"\b\w+\b", text.lower())

    if not words:
        return False, text

    # tokenize reversed text
    reversed_text = text[::-1]
    reversed_words = re.findall(r"\b\w+\b", reversed_text.lower())

    # score both
    original_score = sum(w in common_words for w in words)
    reversed_score = sum(w in common_words for w in reversed_words)

    # normalize
    original_ratio = original_score / max(len(words), 1)
    reversed_ratio = reversed_score / max(len(reversed_words), 1)

    if reversed_ratio > original_ratio and reversed_ratio >= threshold:
        return True, reversed_text

    return False, text

# def run_agent(question: str) -> str:
#     """
#     Public API for the agent.
#     Takes a question string and returns an answer string.
#     """
#     graph = build_graph()

#     flag, question_processed = is_reversed_question(question)
#     messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": question_processed}]

#     result = graph.invoke(
#         {"messages": messages}
#     )
    
#     final_answer = extract_final_answer(result["messages"])
#     return final_answer


def run_agent(question: str) -> str:
    """
    Run agent with verbose debugging.
    """

    graph = build_graph(question)

    flag, question_processed = is_reversed_question(question)

    if flag:
        print("\n[DEBUG] Reversed question detected")
        print("[DEBUG] Corrected Question:")
        print(question_processed)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_processed},
    ]

    messages = messages[-6:]

    print("\n================ AGENT RUN ================\n")

    final_messages = None

    for event in graph.stream(
        {"messages": messages},
        stream_mode="values",
        config={"recursion_limit": 5}
    ):

        final_messages = event["messages"]

        latest = final_messages[-1]

        print(f"\n[{latest.type.upper()} MESSAGE]\n")

        # AI message
        if latest.type == "ai":

            # tool calls
            if hasattr(latest, "tool_calls") and latest.tool_calls:

                print("Tool Calls:")

                for tc in latest.tool_calls:
                    print(f"  -> Tool: {tc['name']}")
                    print(f"     Args: {tc['args']}")

            # content
            if latest.content:
                print(latest.content)

        # Tool outputs
        elif latest.type == "tool":

            print(f"Tool Name: {latest.name}")
            print("\nTool Output:\n")

            content = str(latest.content)

            # truncate huge outputs
            if len(content) > 1500:
                content = content[:1500] + "\n...[TRUNCATED]..."

            print(content)

        # Human/System
        else:
            print(latest.content)

    print("\n===========================================\n")

    final_answer = extract_final_answer(final_messages)

    print("[FINAL ANSWER]")
    print(final_answer)

    return final_answer

# test
if __name__ == "__main__":
    first_query = "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations."
    answer = run_agent(first_query)
    print("🎩 Agent's Response:")
    print(answer)