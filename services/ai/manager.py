import os
from services.ai.deterministic_provider import DeterministicProvider
from services.ai.gemini_provider import GeminiProvider
from services.ai.ollama_provider import OllamaProvider
from services.ai.openai_provider import OpenAIProvider
from services.ai.schemas import ProviderStatus, StructuredAnalysisContext


class AIManager:
    """Uses exactly the selected provider; failure falls back locally only."""
    def __init__(self, provider=None):
        name = (provider or os.getenv("AI_PROVIDER", "deterministic")).strip().lower()
        providers = {"deterministic": DeterministicProvider, "ollama": OllamaProvider,
                     "gemini": GeminiProvider, "openai": OpenAIProvider}
        if name not in providers: raise ValueError(f"Unsupported AI_PROVIDER: {name}")
        self.provider_name = name
        self.provider = providers[name]()
        self.fallback = DeterministicProvider()
        self.last_status = ProviderStatus(name, name == "deterministic")

    def _configured(self):
        if self.provider_name == "deterministic": return True
        if self.provider_name == "ollama": return bool(self.provider.model)
        if self.provider_name == "gemini": return bool(self.provider.api_key and self.provider.model)
        return bool(self.provider.api_key and self.provider.model)

    def interpret(self, context: StructuredAnalysisContext):
        failure = None
        for _attempt in range(2):
            try:
                insight = self.provider.generate(context)
                self.last_status = ProviderStatus(self.provider_name, True, True, True, True, False, "")
                return insight
            except Exception as exc:
                failure = exc
        # Deliberately do not include exception text; SDK errors can contain request details.
        self.last_status = ProviderStatus(self.provider_name, self._configured(), False, False, False, True,
                                          f"Configured provider failed ({type(failure).__name__}); local fallback active")
        fallback = self.fallback.generate(context)
        fallback.limitations.append("Configured AI provider failed after one retry; deterministic local reasoning is active.")
        return fallback

    def health(self, generation_test=False):
        status = self.provider.health(generation_test)
        if self.provider_name == "deterministic": return status
        status.fallback_active = not status.generation_successful if generation_test else False
        return status
