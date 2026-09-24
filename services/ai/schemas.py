"""Typed contracts for AI interpretation; numerical analysis remains upstream."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class StructuredAnalysisContext:
    facts: Dict[str, Any]
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredAIInsight:
    summary: str
    key_finding: str
    recommendations: List[str] = field(default_factory=list)
    upgrade_opportunity: str = ""
    incentive_explanation: str = ""
    action_summary: str = ""
    confidence: str = "low"
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "StructuredAIInsight":
        required = {"summary", "key_finding", "recommendations", "upgrade_opportunity",
                    "incentive_explanation", "action_summary", "confidence", "limitations"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("AI output fields do not match the structured insight schema")
        for name in ("summary", "key_finding", "upgrade_opportunity", "incentive_explanation", "action_summary", "confidence"):
            if not isinstance(value[name], str):
                raise ValueError(f"{name} must be text")
        for name in ("recommendations", "limitations"):
            if not isinstance(value[name], list) or any(not isinstance(x, str) for x in value[name]):
                raise ValueError(f"{name} must be a list of text")
        if value["confidence"] not in {"low", "medium", "high"}:
            raise ValueError("confidence must be low, medium, or high")
        return cls(**value)


@dataclass
class ProviderStatus:
    provider: str
    configured: bool
    reachable: bool = False
    model_available: bool = False
    generation_successful: bool = False
    fallback_active: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
