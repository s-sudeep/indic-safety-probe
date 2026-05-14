"""Command-line interface for indic-safety-probe.

Usage:
    indic-safety-probe --model sarvam-105b
    indic-safety-probe --model sarvam-m --probes bec
    indic-safety-probe --model sarvam-30b --probes bec honorifics --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from .core import Suite
from .adapters.sarvam import SarvamCloudAdapter
from .probes.bec import BECProbe
from .probes.honorifics import HonorificsProbe


AVAILABLE_PROBES = {
    "bec": BECProbe,
    "honorifics": HonorificsProbe,
}

SARVAM_MODELS = ["sarvam-m", "sarvam-30b", "sarvam-105b"]


def _print_probe_result(probe_result) -> None:
    """Print one probe's results to stdout. Reuses the format from quickstart.py."""
    print(f"\nProbe: {probe_result.probe_name}")
    for prompt, s in zip(probe_result.prompts, probe_result.scores):
        # Honorifics has nested category info; BEC uses category directly
        category = prompt.category
        register = prompt.metadata.get("register", "") if prompt.metadata else ""
        if register:
            category_str = f"{prompt.metadata['axis']:<20} {register:<8}"
        else:
            category_str = f"{category:<12}"

        score_val = s.value
        if isinstance(score_val, float) and score_val != score_val:  # NaN
            score_str = "  nan"
        else:
            score_str = f"{score_val:.2f}"

        print(
            f"  {prompt.language:>3} {category_str} "
            f"score={score_str}  label={s.label}"
        )

    # Probe-level summary if present (Honorifics fills this)
    if probe_result.summary:
        summ = probe_result.summary
        print(f"\n  --- Probe-level analysis ---")
        for k, v in summ.items():
            if k in ("pair_details", "note"):
                continue
            if isinstance(v, float):
                print(f"  {k:<32} {v:.3f}")
            else:
                print(f"  {k:<32} {v}")

        # Pair details for Honorifics
        if "pair_details" in summ and summ["pair_details"]:
            print(f"\n  Pair breakdown:")
            for pd in summ["pair_details"]:
                marker = " <-asym" if pd.get("asymmetric") else ""
                print(
                    f"    {pd['pair_key']:<32} "
                    f"f={pd['formal_label']:<14} i={pd['informal_label']:<14} "
                    f"lenD={pd['length_ratio']:.2f} lexD={pd['lexical_divergence']:.2f}"
                    f"{marker}"
                )


def _print_stripped_answers(probe_result) -> None:
    """Print the post-<think> answers for each prompt."""
    think_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    print("\n\n=== STRIPPED ANSWERS ===")
    for prompt, s in zip(probe_result.prompts, probe_result.scores):
        answer = think_re.sub("", s.raw_response or "").strip()
        register = (prompt.metadata or {}).get("register", "")
        suffix = f" / {register}" if register else ""
        print(f"\n--- {prompt.language} / {prompt.category}{suffix} "
              f"-> {s.label} ({s.value if isinstance(s.value, float) and s.value == s.value else 'nan'}) ---")
        print(f"[answer: {len(answer)} chars | raw: {len(s.raw_response or '')} chars]")
        print(answer[:600])
        if len(answer) > 600:
            print("...")


def _cache_to_json(result, model_name: str, output_dir: Path) -> Path:
    """Save the run to a JSON file. Returns the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace(":", "_")
    probe_names = "-".join(pr.probe_name for pr in result.probe_results)
    # Keep filenames short
    if len(probe_names) > 50:
        probe_names = f"{len(result.probe_results)}probes"
    path = output_dir / f"{safe_name}_{probe_names}.json"

    payload = {
        "model": result.model_name,
        "timestamp": result.timestamp.isoformat(),
        "probes": [
            {
                "probe_name": pr.probe_name,
                "mean_score": pr.mean_score,
                "failure_rate": pr.failure_rate,
                "summary": pr.summary,
                "results": [
                    {
                        "language": p.language,
                        "category": p.category,
                        "metadata": p.metadata,
                        "prompt_text": p.text,
                        "score_value": (
                            None
                            if isinstance(s.value, float) and s.value != s.value
                            else s.value
                        ),
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indic-safety-probe",
        description=(
            "Black-box safety eval for Indic LLMs. Runs probes against a "
            "Sarvam Cloud model and prints results to the terminal."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SARVAM_MODELS,
        help="Sarvam model to evaluate.",
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=list(AVAILABLE_PROBES.keys()),
        default=list(AVAILABLE_PROBES.keys()),
        help="Which probes to run. Default: all.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to inject (e.g., agent context).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per response. Reasoning models need >=2048.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save the JSON cache. Default: ./results",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip saving the JSON cache.",
    )
    parser.add_argument(
        "--show-responses",
        action="store_true",
        help="Print the stripped raw responses after the score table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.environ.get("SARVAM_API_KEY"):
        print("ERROR: SARVAM_API_KEY not set. Add it to .env or export it.",
              file=sys.stderr)
        return 1

    print(f"Initializing {args.model}...")
    model = SarvamCloudAdapter(
        model=args.model,
        system_prompt=args.system_prompt,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    probes = [AVAILABLE_PROBES[name]() for name in args.probes]
    suite = Suite(probes)
    result = suite.run(model, verbose=True)

    print(f"\n=== {model.name} ===")
    for probe_result in result.probe_results:
        _print_probe_result(probe_result)

    if args.show_responses:
        for probe_result in result.probe_results:
            _print_stripped_answers(probe_result)

    if not args.no_cache:
        path = _cache_to_json(result, model.name, args.output_dir)
        print(f"\n[cached: {path}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())