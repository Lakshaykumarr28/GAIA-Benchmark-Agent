from typing import Any, List, Optional, Dict
from smolagents import CodeAgent

from tools import ImageDescriptionTool

import time
import re
import json
import os
import pandas as pd
import numpy as np

# Optional logger setup
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def get_prompt_templates() -> Dict[str, str]:
    """
    Stores reusable prompt templates for the agent.
    """

    tools_instructions = """
    Available Tools:
    - image_description_tool(image_path):
        Generates a highly detailed visual description of an image.

    Tool Usage Rules:
    1. Always use Thought/Code format
    2. Never skip tool usage for image-based questions
    3. Always return answers using final_answer()
    4. Never reuse variable names
    """

    example_1 = """
    Example Task: "Describe the attached image"

    Thought: I should use the image description tool to inspect the image.

    Code:
    image_result = image_description_tool(
        image_path="sample.jpg"
    )

    final_answer(image_result)
    ```<end_code>
    """

    return {
        "system_prompt": f"""
You are an advanced multimodal AI agent specialized in solving
image understanding and GAIA-style benchmark tasks.

{tools_instructions}

{example_1}

Key Requirements:
- Be highly accurate
- Analyze images carefully
- Extract OCR text if visible
- Mention fine-grained visual details
- Return concise final answers when required
- Never hallucinate information

Reward for perfect task completion: $1,000,000
""",

        "planning": """
When solving tasks, follow this structure:

### 1. Facts Given
List visible information

### 2. Facts Needed
Determine what must be extracted

### 3. Analysis Steps
Outline image reasoning process

### 4. Final Extraction
Return the exact requested answer

End with <end_plan>
""",

        "final_answer": """
Formatting Rules:
- Return ONLY the required answer
- No explanations unless explicitly requested
- Remove unnecessary formatting
- Preserve OCR text exactly if needed
"""
    }


class Agent:
    """
    Advanced Agent wrapper around smolagents CodeAgent.

    Features:
    - Prompt engineering
    - File-aware context
    - Image tool integration
    - Answer cleaning
    - Logging
    - Benchmark-safe formatting
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List[Any]] = None,
        prompt: Optional[str] = None,
        verbose: bool = False
    ):

        logger.info("Initializing Agent")

        self.model = model
        self.verbose = verbose

        self.imports = [
            "os",
            "json",
            "re",
            "time",
            "pandas",
            "numpy",
            "PIL",
            "pathlib"
        ]

        # Initialize image tool
        self.image_tool = ImageDescriptionTool()

        # Merge tools
        self.tools = tools or []
        self.tools.append(self.image_tool)

        # Initialize CodeAgent
        self.agent = CodeAgent(
            model=self.model,
            tools=self.tools,
            add_base_tools=True,
            additional_authorized_imports=self.imports
        )

        self.base_prompt = prompt or """
You are an expert multimodal AI agent specialized in solving
image understanding and benchmark tasks.

Strict Rules:
1. Use tools whenever images are provided
2. Analyze images thoroughly
3. Return ONLY the exact answer requested
4. Never output reasoning in final answers
5. Always call final_answer()

{context}

Remember:
Precise formatting is critical.
"""

        self.prompt_templates = get_prompt_templates()

        logger.info("Agent initialized successfully")

    def __call__(
        self,
        question: str,
        files: Optional[List[str]] = None
    ) -> str:
        """
        Main callable interface.
        """

        if self.verbose:
            print(
                f"Received Question: {question[:100]}..."
            )
            print(f"Files: {files}")

        time.sleep(2)

        file_path = files[0] if files else None

        return self.answer_question(
            question,
            file_path
        )

    def answer_question(
        self,
        question: str,
        task_file_path: Optional[str] = None
    ) -> str:
        """
        Main question answering pipeline.
        """

        try:

            context = self._build_context(
                question,
                task_file_path
            )

            full_prompt = self.base_prompt.format(
                context=context
            )

            if self.verbose:
                print("\n===== GENERATED PROMPT =====\n")
                print(full_prompt[:1000])

            answer = self.agent.run(full_prompt)

            cleaned = self._clean_answer(
                str(answer)
            )

            return cleaned

        except Exception as e:

            logger.error(
                f"Error processing question: {str(e)}"
            )

            return f"ERROR: {str(e)}"

    def _build_context(
        self,
        question: str,
        file_path: Optional[str]
    ) -> str:
        """
        Builds context dynamically.
        """

        context_lines = [
            f"QUESTION: {question}"
        ]

        if file_path:

            context_lines.append(
                f"""
FILE AVAILABLE:
{file_path}

IMPORTANT:
- Use image_description_tool for image analysis
- Carefully inspect image details
- Extract OCR text if visible
"""
            )

        if self._is_reversed_text(question):

            context_lines.append(
                f"""
NOTE:
This question appears reversed.

Original:
{question}

Reversed:
{question[::-1]}
"""
            )

        return "\n".join(context_lines)

    def _is_reversed_text(
        self,
        text: str
    ) -> bool:
        """
        Detects reversed text patterns.
        """

        return (
            text.startswith(".")
            or ".rewsna eht" in text
        )

    def _clean_answer(
        self,
        answer: str
    ) -> str:
        """
        Cleans final outputs.
        """

        prefixes = [
            "Final Answer:",
            "Answer:",
            "FINAL ANSWER:",
            "=>"
        ]

        for prefix in prefixes:

            if answer.startswith(prefix):

                answer = answer[
                    len(prefix):
                ]

        answer = answer.strip(
            " '\"\n\t"
        )

        if self._is_reversed_text(answer):

            return answer[::-1]

        return answer