from services.ai.base import AIProvider
from services.ai.schemas import ProviderStatus, StructuredAnalysisContext, StructuredAIInsight


class DeterministicProvider(AIProvider):
    name = "deterministic"

    def generate(self, context: StructuredAnalysisContext) -> StructuredAIInsight:
        opportunities = context.evidence
        finding = (opportunities[0].get("title") or opportunities[0].get("reason") or "Review the available energy analysis") if opportunities else "No significant efficiency opportunity was detected from the available data."
        recommendations = [str(item.get("recommended_action", "")) for item in opportunities if item.get("recommended_action")]
        return StructuredAIInsight(
            summary="Analysis is based on the deterministic EnScale energy engine.",
            key_finding=str(finding), recommendations=recommendations[:5],
            upgrade_opportunity="Review equipment only where the deterministic analysis identifies a supported opportunity.",
            incentive_explanation="Incentive matches are reference matches; confirm current terms and eligibility with the program authority.",
            action_summary="Review the evidence and validate proposed operating changes on site.",
            confidence="medium" if opportunities else "low",
            limitations=list(context.limitations),
        )

    def health(self, generation_test: bool = False) -> ProviderStatus:
        return ProviderStatus(self.name, True, True, True, True, False, "Local deterministic reasoning is available")
