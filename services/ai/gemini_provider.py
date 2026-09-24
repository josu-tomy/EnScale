import os
import json
from services.ai.base import AIProvider
from services.ai.remote_provider import _prompt, _validate_numeric_claims
from services.ai.schemas import ProviderStatus, StructuredAnalysisContext, StructuredAIInsight


class GeminiProvider(AIProvider):
    name = "gemini"
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "")
    def generate(self, context):
        from google import genai
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(model=self.model, contents=_prompt(context), config={"response_mime_type": "application/json"})
        return _validate_numeric_claims(StructuredAIInsight.from_dict(json.loads(response.text)), context)
    def health(self, generation_test=False):
        if not (self.api_key and self.model): return ProviderStatus(self.name, False, message="API key or model is not configured")
        try:
            response = self.generate(StructuredAnalysisContext({})) if generation_test else None
            return ProviderStatus(self.name, True, True, True, response is not None, False, "")
        except Exception:
            return ProviderStatus(self.name, True, False, False, False, False, "Configured Gemini service or model is unavailable")
