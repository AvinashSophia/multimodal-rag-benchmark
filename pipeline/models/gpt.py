"""GPT multimodal model wrapper."""

import os
import re
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional
from PIL import Image
from openai import OpenAI
from pipeline.models.base import BaseModel, register_model
from pipeline.utils import ModelResult


@register_model("gpt")
class GPTModel(BaseModel):
    """OpenAI GPT model wrapper for multimodal QA.

    Supports text + image inputs via the GPT-4o API.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_id = config["model"]["gpt"].get("model_id", "gpt-4o")
        self.max_tokens = config["model"]["gpt"].get("max_tokens", 512)
        self.temperature = config["model"]["gpt"].get("temperature", 0.0)

    def _image_to_base64(self, img: Any) -> str:
        """Convert PIL Image to base64 string for API."""
        if isinstance(img, str):
            img = Image.open(img)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_prompt(self, question: str, text_context: List[str], text_ids: Optional[List[str]] = None, has_images: bool = False) -> str:
        """Build the text portion of the prompt. Images are interleaved in run_model."""
        has_text = len(text_context) > 0

        citation_instruction = (
            "Answer concisely with the exact name, number, value, or short phrase as it appears in the source. "
            "Do not write a full sentence. Do not truncate product names, part names, or multi-word values. "
            "Do NOT prefix your answer with 'Answer:' or any label — output the answer directly. "
            "No inline citations or brackets in the answer.\n"
            "On a new line, list ONLY the exact source IDs you used, in this format:\n"
            "Sources: [id1, id2, ...]"
        )

        # Build text context with IDs
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
        """Run GPT on the question with text and image context."""
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

        # Add text prompt
        prompt = self._build_prompt(question, text_context, text_ids=text_ids, has_images=bool(image_context))
        content.append({"type": "text", "text": prompt})

        # Interleave each image with its ID label so GPT can correlate ID → content
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

        # Parse cited source IDs — handles: Sources: [id1, id2], [id1], [id2], or id1, id2
        sources: List[str] = []
        source_match = re.search(r'[Ss]ources:\s*(.+)', answer)
        if source_match:
            raw = source_match.group(1).replace('[', '').replace(']', '')
            sources = [s.strip() for s in raw.split(',') if s.strip()]

        # Strip the entire Sources line so answer metrics aren't polluted by it
        clean_answer = re.sub(r'\s*[Ss]ources:\s*.+', '', answer).strip()

        return ModelResult(
            answer=clean_answer,
            sources=sources,
            raw_response=answer,
            metadata={"model": self.model_id, "usage": dict(response.usage) if response.usage else {}},
        )
