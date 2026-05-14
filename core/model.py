import os
from smolagents import LiteLLMModel


def build_model() -> LiteLLMModel:
    """
    Instantiate a free open-source Hugging Face model
    using Hugging Face Inference API.

    Requires HF_TOKEN in environment variables.
    """

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set. "
            "Add it to your HF Space secrets."
        )

    return LiteLLMModel(
        model_id="huggingface/Qwen/Qwen2.5-7B-Instruct",
        api_key=hf_token
    )