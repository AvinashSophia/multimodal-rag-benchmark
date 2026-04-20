"""Google Gemini via Vertex AI — uses GCP credentials instead of API key."""

import re
from typing import Dict, List, Any, Optional
from PIL import Image
from pipeline.models.base import BaseModel, register_model
from pipeline.utils import ModelResult


@register_model("gemini_vertex")
class GeminiVertexModel(BaseModel):
    """Gemini model wrapper using Vertex AI (GCP credentials).

    Use this instead of GeminiModel when you have GCP credits but have
    exhausted the Gemini API free-tier quota.

    Requires:
        - `gcloud auth application-default login` run once in the terminal, OR
        - GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service account key
        - GCP project with Vertex AI API enabled

    Config keys (under model.gemini_vertex):
        project_id: GCP project ID
        location:   GCP region (default: us-central1)
        model_id:   Gemini model name (default: gemini-2.0-flash)
        max_tokens: Max output tokens (default: 512)
        temperature: Sampling temperature (default: 0.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertex_cfg = config["model"].get("gemini_vertex", {})
        self.project_id = vertex_cfg.get("project_id", "")
        self.location = vertex_cfg.get("location", "us-central1")
        self.model_id = vertex_cfg.get("model_id", "gemini-2.0-flash")
        self.max_tokens = vertex_cfg.get("max_tokens", 512)
        self.temperature = vertex_cfg.get("temperature", 0.0)

        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(
            self.model_id,
            system_instruction=(
                "You are a precise question-answering assistant. "
                "Always follow this exact output format:\n"
                "1. Answer: the exact name, number, value, or short phrase from the source — "
                "no full sentences, no truncation of product names or multi-word values.\n"
                "2. On a new line: Sources: [id1, id2, ...] listing every source ID you used.\n"
                "Never skip the Sources line. Never add explanation or extra text."
            ),
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
        """Run Gemini via Vertex AI with text and image context."""
        from vertexai.generative_models import GenerationConfig, Image as VertexImage
        from io import BytesIO

        prompt = self._build_prompt(question, text_context, text_ids=text_ids, has_images=bool(image_context))

        # Build content list: text prompt, then interleave [image_id]: label + image
        content: List[Any] = [prompt]
        for i, img in enumerate(image_context):
            if isinstance(img, str):
                img = Image.open(img)
            # Convert PIL Image to Vertex AI Image via bytes
            buf = BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            vertex_img = VertexImage.from_bytes(buf.getvalue())
            img_id = image_ids[i] if image_ids and i < len(image_ids) else f"image_{i}"
            content.append(f"[{img_id}]:")
            content.append(vertex_img)

        from vertexai.generative_models import HarmCategory, HarmBlockThreshold, SafetySetting
        response = self.model.generate_content(
            content,
            generation_config=GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
            safety_settings=[
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            ],
        )

        # Gracefully handle blocked responses
        try:
            answer: str = response.text or ""
        except ValueError:
            return ModelResult(
                answer="",
                sources=[],
                raw_response="[BLOCKED by safety filter]",
                metadata={"model": self.model_id, "project": self.project_id, "location": self.location},
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
            metadata={"model": self.model_id, "project": self.project_id, "location": self.location},
            token_usage={"input_tokens": getattr(usage, "prompt_token_count") or 0, "output_tokens": getattr(usage, "candidates_token_count") or 0} if usage else {},
        )
