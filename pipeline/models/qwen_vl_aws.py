"""Qwen-VL via self-hosted OpenAI-compatible endpoint.

The infrastructure team exposes the model at:
    POST https://staging.docu-mind.com/v1/chat/completions
    model name: qwen3-vl-8b-instruct

Uses the standard OpenAI Python client with a custom base_url — no boto3 or
SageMaker SDK required. Multimodal input (text + images) is sent as base64
image_url content blocks, identical to the GPT wrapper.

Config keys (under model.qwen_vl_aws):
    base_url:    Endpoint base URL (default: https://staging.docu-mind.com/v1)
    model_id:    Model name the endpoint expects (default: qwen3-vl-8b-instruct)
    api_key_env: Name of the env var holding the API key (default: QWEN_VL_API_KEY)
                 Set to empty string or "none" if the endpoint requires no auth.
    max_tokens:  Max output tokens (default: 1024)
    temperature: Sampling temperature (default: 0.0)
"""

import os
import re
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional

from PIL import Image
from openai import OpenAI

from pipeline.models.base import BaseModel, register_model
from pipeline.utils import ModelResult


@register_model("qwen_vl_aws")
class QwenVLAWSModel(BaseModel):
    """Qwen-VL wrapper for the self-hosted OpenAI-compatible endpoint."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cfg = config["model"].get("qwen_vl_aws", {})

        self.model_id = cfg.get("model_id", "qwen3-vl-8b-instruct")
        self.max_tokens = cfg.get("max_tokens", 1024)
        self.temperature = cfg.get("temperature", 0.0)

        base_url = cfg.get("base_url", "https://staging.docu-mind.com/v1")
        api_key_env = cfg.get("api_key_env", "QWEN_VL_API_KEY")
        api_key = os.getenv(api_key_env) or "none"

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _image_to_base64(self, img: Any) -> str:
        if isinstance(img, str):
            img = Image.open(img)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_prompt(self, question: str, text_context: List[str], text_ids: Optional[List[str]] = None, has_images: bool = False) -> str:
        has_text = len(text_context) > 0

        citation_instruction = (
            "Answer concisely with the exact name, number, value, or short phrase as it appears in the source. "
            "Do not write a full sentence. Do not truncate product names, part names, or multi-word values. "
            "Do NOT prefix your answer with 'Answer:' or any label — output the answer directly. "
            "No inline citations or brackets in the answer.\n"
            "On a new line, list ONLY the exact source IDs you used, in this format:\n"
            "Sources: [id1, id2, ...]"
        )

        if has_text:
            if text_ids:
                context_str = "\n\n".join(
                    [f"[{text_ids[i]}]: {chunk}" for i, chunk in enumerate(text_context)]
                )
            else:
                context_str = "\n\n".join(
                    [f"[doc_{i}]: {chunk}" for i, chunk in enumerate(text_context)]
                )

        if has_text and has_images:
            return (
                f"Answer the following question based on the provided text and labeled images below.\n\n"
                f"Text Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                f"Each image is preceded by its ID label (e.g. [image_id]:).\n\n"
                f"{citation_instruction}"
            )
        elif has_images:
            return (
                f"Answer the following question based on the labeled images below.\n\n"
                f"Question: {question}\n\n"
                f"Each image is preceded by its ID label (e.g. [image_id]:).\n\n"
                f"{citation_instruction}"
            )
        else:
            return (
                f"Answer the following question based on the provided context.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                f"{citation_instruction}"
            )

    def run_model(
        self,
        question: str,
        text_context: List[str],
        image_context: List[Any],
        text_ids: Optional[List[str]] = None,
        image_ids: Optional[List[str]] = None,
    ) -> ModelResult:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering assistant. "
                    "Always follow this exact output format:\n"
                    "1. The exact name, number, value, or short phrase from the source — "
                    "output it directly with no label or prefix, no full sentences, "
                    "no truncation of product names or multi-word values.\n"
                    "2. On a new line: Sources: [id1, id2, ...] listing every source ID you used.\n"
                    "Never skip the Sources line. Never add explanation or extra text. "
                    "Never prefix the answer with 'Answer:' or any other label."
                ),
            }
        ]

        content: List[Dict[str, Any]] = []
        prompt = self._build_prompt(question, text_context, text_ids=text_ids, has_images=bool(image_context))
        content.append({"type": "text", "text": prompt})

        for i, img in enumerate(image_context):
            img_id = image_ids[i] if image_ids and i < len(image_ids) else f"image_{i}"
            b64 = self._image_to_base64(img)
            content.append({"type": "text", "text": f"[{img_id}]:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer = response.choices[0].message.content or ""

        # Parse cited source IDs
        sources: List[str] = []
        source_match = re.search(r'[Ss]ources:\s*(.+)', answer)
        if source_match:
            raw = source_match.group(1).replace('[', '').replace(']', '')
            sources = [s.strip() for s in raw.split(',') if s.strip()]

        clean_answer = re.sub(r'\s*[Ss]ources:\s*.+', '', answer).strip()

        usage = response.usage
        return ModelResult(
            answer=clean_answer,
            sources=sources,
            raw_response=answer,
            metadata={"model": self.model_id},
            token_usage={"input_tokens": usage.prompt_tokens or 0, "output_tokens": usage.completion_tokens or 0} if usage else {},
        )
