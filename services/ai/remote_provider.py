"""Small standard-library HTTP adapter for Ollama and OpenAI-compatible APIs."""
import json
import urllib.error
import urllib.request

from services.ai.schemas import StructuredAIInsight

INSIGHT_KEYS = ["summary", "key_finding", "recommendations", "upgrade_opportunity",
                "incentive_explanation", "action_summary", "confidence", "limitations"]


def _request(url, payload=None, headers=None, timeout=15, method="POST"):
    data = json.dumps(payload or {}).encode() if method != "GET" else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _prompt(context):
    return ("Interpret this EnScale structured energy analysis. Do not calculate or introduce any numerical "
            "facts, dates, savings, ratings, payback, emissions, or incentive values. Use only supplied facts. "
            "Return only JSON with exactly these fields: " + ", ".join(INSIGHT_KEYS) +
            ". Recommendations and limitations must be arrays of strings; confidence is low, medium, or high.\n" +
            json.dumps(context.to_dict(), ensure_ascii=False))


def _validate_numeric_claims(insight, context):
    """Reject numeric tokens not present in deterministic source facts."""
    import re
    text = json.dumps(insight.to_dict(), ensure_ascii=False)
    source = json.dumps(context.to_dict(), ensure_ascii=False)
    numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*%?", text))
    allowed = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*%?", source))
    if numbers - allowed:
        raise ValueError("AI output introduced numerical claims absent from deterministic facts")
    return insight


class OllamaClient:
    def __init__(self, base_url, model): self.base_url, self.model = base_url.rstrip("/"), model
    def generate(self, context):
        result = _request(self.base_url + "/api/generate", {"model": self.model, "prompt": _prompt(context), "stream": False, "format": "json"})
        return _validate_numeric_claims(StructuredAIInsight.from_dict(json.loads(result["response"])), context)
    def health(self, generation_test=False):
        models = _request(self.base_url + "/api/tags", method="GET")
        available = any(m.get("name", "").split(":")[0] == self.model.split(":")[0] for m in models.get("models", []))
        return models, available


class OpenAICompatibleClient:
    def __init__(self, base_url, model, api_key=""):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
    def generate(self, context):
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        result = _request(self.base_url + "/chat/completions", {"model": self.model, "messages": [{"role": "user", "content": _prompt(context)}], "response_format": {"type": "json_object"}}, headers)
        raw = result["choices"][0]["message"]["content"]
        return _validate_numeric_claims(StructuredAIInsight.from_dict(json.loads(raw)), context)
