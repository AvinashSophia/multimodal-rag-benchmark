"""Google Gemini multimodal model wrapper using the google-genai SDK."""

import os
import re
from io import BytesIO
from typing import Dict, List, Any, Optional
from PIL import Image
from pipeline.models.base import BaseModel, register_model
from pipeline.utils import ModelResult


@register_model("gemini")
class GeminiModel(BaseModel):
    """Google Gemini model wrapper for multimodal QA.

    Uses the new google-genai SDK (replaces deprecated google-generativeai).
    Supports text + image inputs via the Gemini API.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from google import genai
        from google.genai import types

        self.model_id = config["model"]["gemini"].get("model_id", "gemini-2.0-flash")
        self.max_tokens = config["model"]["gemini"].get("max_tokens", 512)
        self.temperature = config["model"]["gemini"].get("temperature", 0.0)

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.generation_config = types.GenerateContentConfig(
            system_instruction=(
                "You are a precise question-answering assistant. "
                "Always follow this exact output format:\n"
                "1. Answer: the exact name, number, value, or short phrase from the source — "
                "no full sentences, no truncation of product names or multi-word values.\n"
                "2. On a new line: Sources: [id1, id2, ...] listing every source ID you used.\n"
                "Never skip the Sources line. Never add explanation or extra text."
            ),
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ],
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
        """Run Gemini on the question with text and image context."""
        from google.genai import types

        prompt = self._build_prompt(question, text_context, text_ids=text_ids, has_images=bool(image_context))

        # Build content: text prompt, then interleave [image_id]: label + image bytes
        contents: List[Any] = [prompt]
        for i, img in enumerate(image_context):
            if isinstance(img, str):
                img = Image.open(img)
            buf = BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            img_id = image_ids[i] if image_ids and i < len(image_ids) else f"image_{i}"
            contents.append(f"[{img_id}]:")
            contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=self.generation_config,
        )

        # Gracefully handle blocked responses
        try:
            answer: str = response.text or ""
        except ValueError:
            return ModelResult(
                answer="",
                sources=[],
                raw_response="[BLOCKED by safety filter]",
                metadata={"model": self.model_id},
                token_usage={},
            )

        # Parse cited source IDs — handles wrapped IDs (Gemini may insert newlines mid-ID)
        # [\s\S]+ captures across newlines; re.sub(\s+) collapses any mid-ID whitespace
        sources: List[str] = []
        source_match = re.search(r'[Ss]ources:\s*([\s\S]+)', answer)
        if source_match:
            raw = source_match.group(1).replace('[', '').replace(']', '')
            sources = [re.sub(r'\s+', '', s) for s in raw.split(',') if s.strip()]

        # Strip the entire Sources section (may span multiple lines) from the answer
        clean_answer = re.sub(r'\s*[Ss]ources:\s*[\s\S]+', '', answer).strip()

        usage = getattr(response, "usage_metadata", None)
        return ModelResult(
            answer=clean_answer,
            sources=sources,
            raw_response=answer,
            metadata={"model": self.model_id},
            token_usage={"input_tokens": getattr(usage, "prompt_token_count") or 0, "output_tokens": getattr(usage, "candidates_token_count") or 0} if usage else {},
        )
