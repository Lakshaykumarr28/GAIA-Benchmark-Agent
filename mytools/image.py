import os
import base64
from pathlib import Path

from smolagents import Tool


class ImageDescriptionTool(Tool):
    """
    Describes an image using the HF Inference API (Qwen2.5-VL via InferenceClient).
    No local model is loaded — works on CPU-only HF Spaces.
    """

    name = "describe_image"
    description = (
        "Analyzes an image file and returns a detailed description including "
        "objects, text/OCR, numbers, colors, layout, and any data visible. "
        "Pass the local file path returned by download_task_file."
    )
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Local path to the image file.",
        }
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        # Strip any prefix message from DownloadTaskFileTool
        if "Image file saved to:" in image_path:
            image_path = image_path.split("Image file saved to:")[-1].split("\n")[0].strip()

        if not os.path.exists(image_path):
            return f"[ImageDescriptionTool ERROR] File not found: {image_path}"

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            return "[ImageDescriptionTool ERROR] HF_TOKEN not set"

        try:
            from huggingface_hub import InferenceClient
            from PIL import Image
            import io

            # Read and encode image
            img = Image.open(image_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            b64 = base64.b64encode(img_bytes).decode("utf-8")

            client = InferenceClient(token=hf_token)

            prompt = (
                "Describe this image in detail. Include:\n"
                "1. All visible text and numbers (OCR)\n"
                "2. Objects, people, animals present\n"
                "3. Charts/tables/data if any — list all values\n"
                "4. Colors, layout, spatial relationships\n"
                "5. Any contextual details useful for answering questions\n\n"
                "Be precise and exhaustive."
            )

            response = client.chat_completion(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content

        except Exception as exc:
            return f"[ImageDescriptionTool ERROR] {exc}"
