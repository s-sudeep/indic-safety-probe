# indic-safety-probe

Black-box safety eval for Indic LLMs. Sends adversarial prompts to a hosted Indic model, scores the responses, and saves audit-grade JSON.

```
pip install indic-safety-probe
export SARVAM_API_KEY=...
indic-safety-probe --model sarvam-105b
```

## What it does

Two probes, run against any Sarvam Cloud model:

- **BECProbe** — Tests CFO-impersonation BEC fraud-email generation in English, Hindi, Kannada, Bengali, Tamil, Telugu. Each harmful prompt has a structurally matched benign control.
- **HonorificsProbe** — Tests whether formal vs. informal honorific register (आप vs तू, etc.) produces asymmetric refusal behavior on identical content requests across Hindi, Tamil, Telugu, Bengali.

Results print to terminal; JSON cached to `./results/`.

## Key finding

Sarvam's API produces complete, well-formed BEC fraud emails in all 6 tested languages across `sarvam-m`, `sarvam-30b`, and `sarvam-105b`. Benign controls (legitimate quarterly vendor payments with identical structure) produce appropriate output. The model distinguishes the cases. It does not refuse the harmful one. The same English prompts have been observed to refuse on the Sarvam dashboard — suggesting safety lives in the deployment surface, not the model. Deployers calling the API directly inherit none of the dashboard's apparent safety.

Register variation (formal vs informal honorific) does not act as a jailbreak vector.

Cached results in `results/`.

## Usage

```
indic-safety-probe --model sarvam-m
indic-safety-probe --model sarvam-105b --probes bec
indic-safety-probe --model sarvam-30b --system-prompt "You are a customer service agent for Indian Railways."
```

## Roadmap

v0.2 adds: hallucination under regulatory framing, implicit bias by caste/gender, PII regurgitation, hate-speech generation, LLM-as-judge scoring, additional vendor adapters (Krutrim, CoRover).

Out of scope: adversarial suffix attacks (white-box), agent-framework probes (orchestration-layer), CBRN content.

## Note on development

This is a vibe-coded v0.1 built in ~14 hours for an Activate AI Fellows submission. Code is short (~900 lines, half prompts). Findings are real and reproducible from the cached JSON. The design decisions — probe taxonomy, benign-control structure, reasoning-trace handling, scope boundaries — are author-owned; the implementation was co-written with an LLM.

## License

MIT. See `LICENSE`.

## Author

Sudeep — https://github.com/s-sudeep/indic-safety-probe

