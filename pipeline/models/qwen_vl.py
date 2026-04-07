"""Qwen-VL multimodal model wrapper."""

import re
from typing import Dict, List, Any, Optional
from PIL import Image
from pipeline.models.base import BaseModel, register_model
from pipeline.utils import ModelResult


@register_model("qwen_vl")
class QwenVLModel(BaseModel):
    """Qwen-VL model wrapper for multimodal QA.

    Runs locally on GPU. This is Sophia's production model.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model_id = config["model"]["qwen_vl"].get(
            "model_id", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self.max_tokens = config["model"]["qwen_vl"].get("max_tokens", 512)

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto"
        )

    def _build_prompt(self, question: str, text_context: List[str], text_ids: Optional[List[str]] = None, has_images: bool = False) -> str:
        """Build the text portion of the prompt. Images are interleaved in run_model."""
        has_text = len(text_context) > 0

        citation_instruction = (
            "Answer concisely with the exact name, number, value, or short phrase as it appears in the source. "
            "Do not write a full sentence. Do not truncate product names, part names, or multi-word values. "
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
        """Run Qwen-VL on the question with text and image context."""
        import torch

        prompt = self._build_prompt(question, text_context, text_ids=text_ids, has_images=bool(image_context))

        # Build message content: interleave [image_id]: label + image before the text prompt
        content: List[Dict[str, Any]] = []
        for i, img in enumerate(image_context):
            if isinstance(img, str):
                img = Image.open(img)
            img_id = image_ids[i] if image_ids and i < len(image_ids) else f"image_{i}"
            content.append({"type": "text", "text": f"[{img_id}]:"})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering assistant. "
                    "Always follow this exact output format:\n"
                    "1. Answer: the exact name, number, value, or short phrase from the source — "
                    "no full sentences, no truncation of product names or multi-word values.\n"
                    "2. On a new line: Sources: [id1, id2, ...] listing every source ID you used.\n"
                    "Never skip the Sources line. Never add explanation or extra text."
                ),
            },
            {"role": "user", "content": content},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        pil_images = [img for img in image_context if isinstance(img, Image.Image)]
        inputs = self.processor(
            text=[text_input],
            images=pil_images if pil_images else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(  # type: ignore[misc]
                **inputs, max_new_tokens=self.max_tokens
            )

        # Decode only the generated tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

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
            metadata={"model": self.model_id},
        )
