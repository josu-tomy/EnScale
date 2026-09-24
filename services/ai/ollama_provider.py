import os
from services.ai.base import AIProvider
from services.ai.remote_provider import OllamaClient
from services.ai.schemas import ProviderStatus, StructuredAnalysisContext


class OllamaProvider(AIProvider):
    name = "ollama"
    def __init__(self, base_url=None, model=None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "")
    def _client(self): return OllamaClient(self.base_url, self.model)
    def generate(self, context): return self._client().generate(context)
    def health(self, generation_test=False):
        configured = bool(self.model)
        try:
            _, available = self._client().health()
            successful = bool(self.generate(StructuredAnalysisContext({})) if generation_test and available else False)
            return ProviderStatus(self.name, configured, True, available, successful, False, "")
        except Exception:
            return ProviderStatus(self.name, configured, False, False, False, False, "Ollama endpoint or configured model is unavailable")
