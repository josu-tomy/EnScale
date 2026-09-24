# EnScale AI Setup

The AI layer interprets deterministic EnScale results. Numeric energy, cost, savings, emissions, forecasts, constraints, and incentive values continue to come from the existing engine. Failed configured providers fall back to local deterministic interpretation without switching to another cloud service. Set variables in your shell or a local `.env` file (never commit `.env`).

## A. Deterministic mode (default)

```bash
AI_PROVIDER=deterministic
python scripts/check_ai.py
```

This mode is offline and requires no key or model.

## B. Ollama local mode

Install and start Ollama using its official installation instructions, then pull a model supported by your machine:

```bash
ollama serve
ollama pull llama3.1
export AI_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1
python scripts/check_ai.py
```

Troubleshooting: check that the service is listening at `OLLAMA_BASE_URL`, that the model name matches `ollama list`, and that the selected model can return JSON. Failure leaves deterministic fallback active.

## C. Gemini API mode

Install the current Google Gen AI SDK and configure a key/model from Google AI Studio:

```bash
pip install google-genai
export AI_PROVIDER=gemini
export GEMINI_API_KEY='your-key-in-your-shell-only'
export GEMINI_MODEL='your-enabled-model'
python scripts/check_ai.py
```

Troubleshooting: confirm key access, model availability, network access, and SDK installation. Credentials are never included in diagnostic output.

## D. OpenAI-compatible mode

```bash
export AI_PROVIDER=openai
export OPENAI_API_KEY='your-key-in-your-shell-only'
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL='your-enabled-model'
python scripts/check_ai.py
```

For a compatible endpoint, set `OPENAI_BASE_URL` to its `/v1` API root and supply that service's key/model. This provider is selected explicitly; failures do not trigger another cloud provider.

## Health fields

`scripts/check_ai.py` prints configured, reachable, model availability, generation test, and fallback status. Provider exceptions and secrets are not printed. A successful health probe may make one small generation request for remote providers.
