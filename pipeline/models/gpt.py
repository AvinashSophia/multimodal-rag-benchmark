"""GPT multimodal model wrapper."""

import os
import base64
from io import BytesIO
from typing import Dict, List, Any
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

    def _build_prompt(self, question: str, text_context: List[str],text_ids: List[str] = None,image_ids: List[str] = None) -> str:
        
        has_text = len(text_context) > 0
        has_images = image_ids is not None and len(image_ids) > 0
        
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

        # Build image reference with IDs
        if has_images:
            image_ref_str = ", ".join(image_ids)

         # Build prompt based on available data
        if has_text and has_images:
            return (
                f"Answer the following question based on the provided text and images.\n\n"
                f"Text Context:\n{context_str}\n\n"
                f"Image IDs for reference: {image_ref_str}\n\n"
                f"Question: {question}\n\n"
                f"Provide a concise answer. Then list the exact source IDs you used in this format:\n"
                f"Sources: [id1, id2, ...]"
            )
        elif has_images:
            return (
                f"Answer the following question based on the provided images.\n\n"
                f"Image IDs for reference: {image_ref_str}\n\n"
                f"Question: {question}\n\n"
                f"Provide a concise answer. Then list the exact image IDs you used in this format:\n"
                f"Sources: [id1, id2, ...]"
            )
        else:
            return (
                f"Answer the following question based on the provided context.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                f"Provide a concise answer. Then list the exact source IDs you used in this format:\n"
                f"Sources: [id1, id2, ...]"
            )

    def run_model(
        self,
        question: str,
        text_context: List[str],
        image_context: List[Any],
        text_ids: List[str] = None,
        image_ids: List[str] = None
    ) -> ModelResult:
        """Run GPT on the question with text and image context."""
        messages = []
        content = []

        # Add text prompt
        prompt = self._build_prompt(question, text_context,text_ids=text_ids, image_ids=image_ids)
        content.append({"type": "text", "text": prompt})

        # Add images if present
        for img in image_context:
            b64 = self._image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer = response.choices[0].message.content

        return ModelResult(
            answer=answer,
            sources=[],
            raw_response=answer,
            metadata={"model": self.model_id, "usage": dict(response.usage) if response.usage else {}},
        )
