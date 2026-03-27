"""Google Gemini multimodal model wrapper."""

import os
from typing import Dict, List, Any
from PIL import Image
from pipeline.models.base import BaseModel, register_model
from pipeline.utils import ModelResult


@register_model("gemini")
class GeminiModel(BaseModel):
    """Google Gemini model wrapper for multimodal QA.

    Supports text + image inputs via the Gemini API.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_id = config["model"]["gemini"].get("model_id", "gemini-2.0-flash")
        self.max_tokens = config["model"]["gemini"].get("max_tokens", 512)
        self.temperature = config["model"]["gemini"].get("temperature", 0.0)
        self.model = genai.GenerativeModel(self.model_id)

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
        """Run Gemini on the question with text and image context."""
        prompt = self._build_prompt(question, text_context, has_images=len(image_context) > 0)

        # Build content list with text and images
        content = [prompt]
        for img in image_context:
            if isinstance(img, str):
                img = Image.open(img)
            content.append(img)

        response = self.model.generate_content(
            content,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        answer = response.text

        return ModelResult(
            answer=answer,
            sources=[f"doc_{i+1}" for i in range(len(text_context))]
            + [f"img_{i+1}" for i in range(len(image_context))],
            raw_response=answer,
            metadata={"model": self.model_id},
        )
