import os
from services.ai.base import AIProvider
from services.ai.remote_provider import OpenAICompatibleClient
from services.ai.schemas import ProviderStatus, StructuredAnalysisContext


class OpenAIProvider(AIProvider):
    name = "openai"
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "")
    def _client(self): return OpenAICompatibleClient(self.base_url, self.model, self.api_key)
    def generate(self, context): return self._client().generate(context)
    def health(self, generation_test=False):
        configured = bool(self.api_key and self.model)
        if not configured: return ProviderStatus(self.name, False, message="API key or model is not configured")
        try:
            result = self.generate(StructuredAnalysisContext({})) if generation_test else None
            return ProviderStatus(self.name, True, True, True, result is not None, False, "")
        except Exception:
            return ProviderStatus(self.name, True, False, False, False, False, "Configured endpoint or model is unavailable")
