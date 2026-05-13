from smolagents import Tool
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

from smolagents import LiteLLMModel


def check_reasoning(final_answer, agent_memory):
    """
    Validates whether the agent reasoning and final answer
    correctly solve the task.
    """

    model_name = "qwen2.5:14b"

    reasoning_model = LiteLLMModel(
        model_id=f"ollama_chat/{model_name}",
        temperature=0.1
    )

    prompt = f"""
You are a strict evaluator for an AI agent.

Below are the agent reasoning steps:

{agent_memory.get_succinct_steps()}

-----------------------------------

FINAL ANSWER:
{final_answer}

Your task:
1. Verify whether the reasoning process is logically correct
2. Verify whether the final answer correctly solves the task
3. Detect hallucinations or unsupported assumptions
4. Detect incomplete reasoning
5. Check whether the final answer contradicts the reasoning

Evaluation Rules:
- Be strict
- Only PASS if confidence > 90%
- FAIL if reasoning is weak, incomplete, or incorrect

Output Format:
1. Reasons for PASS/FAIL
2. Final Decision: PASS or FAIL
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    output = reasoning_model(messages).content

    print("\n===== REASONING CHECK =====\n")
    print(output)

    if "FAIL" in output.upper():
        raise Exception(
            f"Reasoning Validation Failed:\n{output}"
        )

    return True


def ensure_formatting(final_answer, agent_memory):
    """
    Ensures the final answer follows strict formatting rules.
    """

    model_name = "qwen2.5:7b"

    formatting_model = LiteLLMModel(
        model_id=f"ollama_chat/{model_name}",
        flatten_messages_as_text=True,
        temperature=0.0
    )

    prompt = f"""
You are a strict formatting validator for benchmark tasks.

Agent Reasoning:
{agent_memory.get_succinct_steps()}

-----------------------------------

FINAL ANSWER:
{final_answer}

Your task:
Check whether the FINAL ANSWER strictly follows
benchmark formatting requirements.

Formatting Rules:
1. Return ONLY the requested answer
2. No explanations
3. No markdown
4. No bullet points
5. No brackets unless requested
6. Numbers:
   - no commas
   - no units unless specified
   - Arabic numerals only
7. Strings:
   - lowercase when appropriate
   - no articles
   - no abbreviations unless requested
8. Lists:
   - comma separated only
   - no brackets []
9. OCR text must preserve exact spelling if required
10. Remove unnecessary whitespace

Evaluation:
- PASS only if perfectly formatted
- FAIL if any formatting issue exists

Output Format:
1. Reasons for PASS/FAIL
2. Final Decision: PASS or FAIL
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    output = formatting_model(messages).content

    print("\n===== FORMAT CHECK =====\n")
    print(output)

    if "FAIL" in output.upper():
        raise Exception(
            f"Formatting Validation Failed:\n{output}"
        )

    return True


class ImageDescriptionTool(Tool):
    name = "image_description_tool"
    description = (
        "Loads an image and generates a highly detailed description "
        "including objects, scene understanding, OCR text, layout, "
        "colors, emotions, activities, and contextual reasoning."
    )

    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image file"
        }
    }

    output_type = "string"

    def __init__(self):
        super().__init__()

        print("Loading Qwen2.5-VL-7B-Instruct...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct"
        )

        print("Model loaded successfully.")

    def forward(self, image_path: str) -> str:

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": """
Describe this image in extreme detail.

Your response should include:
1. Overall scene summary
2. Objects and entities present
3. Human appearance and activities
4. Clothing and accessories
5. Facial expressions and emotions
6. Background details
7. Colors and lighting
8. Spatial relationships
9. Visible text/OCR
10. Possible context or story
11. Camera angle and composition
12. Art style or realism
13. Environmental conditions
14. Important fine-grained details

Be exhaustive and precise.
"""
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0]


# Example usage
if __name__ == "__main__":

    tool = ImageDescriptionTool()

    result = tool.forward("sample_image.jpg")

    print("\n===== IMAGE DESCRIPTION =====\n")
    print(result)