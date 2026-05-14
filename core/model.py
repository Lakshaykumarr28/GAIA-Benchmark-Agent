import os
from smolagents import InferenceClientModel


def build_model() -> InferenceClientModel:
    """
    Instantiate a Hugging Face Inference API model via smolagents native client.

    Requires HF_TOKEN in environment variables (set in HF Space secrets).
    Uses Qwen2.5-72B-Instruct — strong enough for GAIA Level 1.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add it to your HF Space secrets."
        )

    return InferenceClientModel(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        token=hf_token,
    )
