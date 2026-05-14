"""Smoke test for HonorificsProbe on sarvam-m."""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import re
import json
from pathlib import Path

from indic_safety_probe import Suite
from indic_safety_probe.adapters.sarvam import SarvamCloudAdapter
from indic_safety_probe.probes.honorifics import HonorificsProbe


def main() -> int:
    if not os.environ.get("SARVAM_API_KEY"):
        print("ERROR: SARVAM_API_KEY not set", file=sys.stderr)
        return 1

    model = SarvamCloudAdapter(
        model="sarvam-105b",
        system_prompt=None,
        temperature=0.0,
        max_tokens=2048,
    )

    suite = Suite([HonorificsProbe()])
    result = suite.run(model)

    print(f"\n=== {model.name} ===")
    for probe_result in result.probe_results:
        print(f"\nProbe: {probe_result.probe_name}")
        for prompt, s in zip(probe_result.prompts, probe_result.scores):
            print(
                f"  {prompt.language:>3} {prompt.metadata['axis']:<20} "
                f"{prompt.metadata['register']:<8} "
                f"score={s.value!s:<6} label={s.label}"
            )

        # Pair-level summary
        summ = probe_result.summary
        print(f"\n  --- Pair-level analysis ({summ['n_pairs']} pairs) ---")
        print(f"  Refusal asymmetry rate: {summ['refusal_asymmetry_rate']:.2f} "
              f"({summ['refusal_asymmetry_pairs']}/{summ['n_pairs']} pairs)")
        print(f"  Mean length ratio:       {summ['mean_length_ratio']:.3f}")
        print(f"  Mean lexical divergence: {summ['mean_lexical_divergence']:.3f}")

        # Per-pair table
        print(f"\n  Pair breakdown:")
        for pd in summ["pair_details"]:
            marker = " ←asym" if pd["asymmetric"] else ""
            print(
                f"    {pd['pair_key']:<32} "
                f"f={pd['formal_label']:<14} i={pd['informal_label']:<14} "
                f"lenΔ={pd['length_ratio']:.2f} lexΔ={pd['lexical_divergence']:.2f}"
                f"{marker}"
            )

        # Stripped answers
        think_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        print("\n\n=== STRIPPED ANSWERS ===")
        for prompt, s in zip(probe_result.prompts, probe_result.scores):
            answer = think_re.sub("", s.raw_response or "").strip()
            print(f"\n--- {prompt.language} / {prompt.metadata['axis']} / "
                  f"{prompt.metadata['register']} → {s.label} ---")
            print(f"[answer length: {len(answer)} chars]")
            print(answer[:600])
            if len(answer) > 600:
                print("...")

    # Cache
    cache_dir = Path("results")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{model.name.replace(':', '_')}_honorifics.json"
    payload = {
        "model": model.name,
        "timestamp": result.timestamp.isoformat(),
        "probes": [
            {
                "probe_name": pr.probe_name,
                "summary": pr.summary,
                "results": [
                    {
                        "language": p.language,
                        "axis": p.metadata["axis"],
                        "register": p.metadata["register"],
                        "pair_key": p.metadata["pair_key"],
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