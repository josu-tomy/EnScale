from abc import ABC, abstractmethod
from typing import Tuple

from services.ai.schemas import ProviderStatus, StructuredAnalysisContext, StructuredAIInsight


class AIProvider(ABC):
    name = "unknown"

    @abstractmethod
    def generate(self, context: StructuredAnalysisContext) -> StructuredAIInsight:
        """Interpret provided facts without calculating or changing them."""

    @abstractmethod
    def health(self, generation_test: bool = False) -> ProviderStatus:
        """Return safe provider health without exposing credentials."""

    def interpret(self, context: StructuredAnalysisContext) -> Tuple[StructuredAIInsight, ProviderStatus]:
        raise NotImplementedError
