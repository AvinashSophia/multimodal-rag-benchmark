"""Qwen-VL multimodal model wrapper."""

from typing import Dict, List, Any
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

    def _build_prompt(self, question: str, text_context: List[str], has_images: bool = False) -> str:
        context_str = "\n\n".join(
            [f"[Document {i+1}]: {chunk}" for i, chunk in enumerate(text_context)]
        )
        
        if has_images and text_context:
            return (
                f"Answer the following question based on the provided text and images.\n\n"
                f"Text Context:\n{context_str}\n\n"
                f"Images are also provided for reference.\n\n"
                f"Question: {question}\n\n"
                f"Provide a concise answer and cite which document(s) or image(s) you used."
            )
        elif has_images:
            return (
                f"Answer the following question based on the provided images.\n\n"
                f"Question: {question}\n\n"
                f"Provide a concise answer and cite which image(s) you used."
            )
        else:
            return (
                f"Answer the following question based on the provided context.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                f"Provide a concise answer and cite which document(s) you used."
            )

    def run_model(
        self,
        question: str,
        text_context: List[str],
        image_context: List[Any],
    ) -> ModelResult:
        """Run Qwen-VL on the question with text and image context."""
        import torch

        prompt = self._build_prompt(question, text_context, has_images=len(image_context) > 0)

        # Build message format for Qwen-VL
        content = []
        for img in image_context:
            if isinstance(img, str):
                content.append({"type": "image", "image": img})
            elif isinstance(img, Image.Image):
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            images=[img for img in image_context if isinstance(img, Image.Image)] or None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens
            )

        # Decode only the generated tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return ModelResult(
            answer=answer,
            sources=[f"doc_{i+1}" for i in range(len(text_context))]
            + [f"img_{i+1}" for i in range(len(image_context))],
            raw_response=answer,
            metadata={"model": self.model_id},
        )
