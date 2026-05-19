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
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from docx import Document
from dotenv import load_dotenv 
# OCR / Vision
from PIL import Image
import pytesseract

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Example: "17 * 42"
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# @tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    content = "\n\n---\n\n".join([f'Doc {i}: \n\n{doc.page_content[:500]}' for i,doc in enumerate(search_docs)])
    return {"wiki_results": content}

# @tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(tavily_api_key=tavily_api_key, max_results=3).invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc["title"]}" page="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}

@tool
def read_pdf(path: str) -> str:
    """Extract text from a PDF file"""
    loader = PyPDFLoader(path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages[:5])  # limit context


@tool
def read_excel(path: str) -> str:
    """Read an Excel file and summarize"""
    df = pd.read_excel(path)
    return df.head(10).to_string()


@tool
def read_csv(path: str) -> str:
    """Read a CSV file"""
    df = pd.read_csv(path)
    return df.head(10).to_string()


@tool
def read_word(path: str) -> str:
    """Read a Word (.docx) file"""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


@tool
def ocr_image(path: str) -> str:
    """Extract text from an image using OCR"""
    image = Image.open(path)
    return pytesseract.image_to_string(image)


# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(system_prompt)

tools = [
    calculator,
    wiki_search,
    web_search,
    read_pdf,
    read_excel,
    read_csv,
    read_word,
    ocr_image,
]
# Build graph function
def build_graph():


    # llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )


    # chat = ChatHuggingFace(llm=llm, verbose=True)
    chat_with_tools = llm.bind_tools(tools)
    # Generate the AgentState and Agent graph
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state: AgentState):
        return {
            "messages": [chat_with_tools.invoke(state["messages"])],
        }
    ## The graph
    builder = StateGraph(AgentState)
    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    #  Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    agent = builder.compile()
    return agent

def extract_final_answer(messages):
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content:
            text = msg.content.strip()
            if text.startswith("FINAL ANSWER:"):
                return text.replace("FINAL ANSWER:", "").strip()
            return text
    return ""

def run_agent(question: str) -> str:
    """
    Public API for the agent.
    Takes a question string and returns an answer string.
    """
    graph = build_graph()
    messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": question}]

    result = graph.invoke(
        {"messages": messages}
    )
    
    final_answer = extract_final_answer(result["messages"])
    return final_answer

# test
if __name__ == "__main__":
    first_query = "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations."
    answer = run_agent(first_query)
    print("🎩 Agent's Response:")
    print(answer)