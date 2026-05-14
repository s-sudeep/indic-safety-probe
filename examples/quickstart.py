"""Smoke test: run BECProbe against a Sarvam model.

Requires:
  uv add openai rich
  export SARVAM_API_KEY=...   (or set in environment)
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys

from indic_safety_probe import Suite
from indic_safety_probe.adapters.sarvam import SarvamCloudAdapter
from indic_safety_probe.probes.bec import BECProbe

import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime


def main() -> int:
    if not os.environ.get("SARVAM_API_KEY"):
        print("ERROR: SARVAM_API_KEY not set", file=sys.stderr)
        return 1

    model = SarvamCloudAdapter(
        model="sarvam-105b",  # start cheap; swap to sarvam-30b / 105b for final cache
        system_prompt=None,
        temperature=0.0,
    )

    suite = Suite([BECProbe()])
    result = suite.run(model)

    # Print a quick text summary; richer terminal output comes later.
    print(f"\n=== {model.name} ===")
    for probe_result in result.probe_results:
        print(f"\nProbe: {probe_result.probe_name}")
        for prompt, s in zip(probe_result.prompts, probe_result.scores):
            print(
                f"  {prompt.language:>3} {prompt.category:<8} "
                f"score={s.value:.2f} label={s.label}"
            )
        harmful_scores = [
            s for p, s in zip(probe_result.prompts, probe_result.scores)
            if p.category == "harmful" and s.label != "error"
        ]
        benign_scores = [
            s for p, s in zip(probe_result.prompts, probe_result.scores)
            if p.category == "benign" and s.label != "error"
        ]
        if harmful_scores:
            avg_harmful = sum(s.value for s in harmful_scores) / len(harmful_scores)
            print(f"\n  Mean failure on harmful: {avg_harmful:.2f}  (higher = worse)")
        if benign_scores:
            avg_benign = sum(s.value for s in benign_scores) / len(benign_scores)
            print(f"  Mean failure on benign:  {avg_benign:.2f}  (high = over-refusal)")

        # --- Stripped-answer debug dump ---
        import re
        think_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        print("\n\n=== STRIPPED ANSWERS (post-</think>) ===")
        for prompt, s in zip(probe_result.prompts, probe_result.scores):
            answer = think_re.sub("", s.raw_response).strip()
            print(f"\n--- {prompt.language} / {prompt.category} → {s.label} ({s.value:.2f}) ---")
            print(f"[answer length: {len(answer)} chars | raw: {len(s.raw_response)} chars]")
            print(answer[:800])
            if len(answer) > 800:
                print("...")

    # Cache the run to disk
    cache_dir = Path("results")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{model.name.replace(':', '_')}_bec.json"

    payload = {
        "model": model.name,
        "timestamp": result.timestamp.isoformat(),
        "probes": [
            {
                "probe_name": pr.probe_name,
                "mean_score": pr.mean_score,
                "failure_rate": pr.failure_rate,
                "results": [
                    {
                        "language": p.language,
                        "category": p.category,
                        "prompt_text": p.text,
                        "score_value": s.value,
                        "score_label": s.label,
                        "explanation": s.explanation,
                        "raw_response": s.raw_response,
                    }
                    for p, s in zip(pr.prompts, pr.scores)
                ],
            }
            for pr in result.probe_results
        ],
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[saved: {cache_path}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())