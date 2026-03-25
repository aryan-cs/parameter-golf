#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from prepare_submission_metadata import merge_summaries, parse_log

ACCEPTANCE_GATE_BPB = 1.1144
RECORD_DIR_REL = Path("records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback")

ARTIFACT_ORDER = [
    "baseline",
    "tttlr25",
    "batch48",
    "tttlr25_batch48",
    "bg3072_tttlr25",
    "chunk16k",
    "epochs2_tttlr25",
    "freeze2_tttlr25",
    "freeze2_epochs2_tttlr25",
    "tttlr30",
]

PROXY_ORDER = [
    "baseline",
    "vr1",
    "bg3072",
    "vr1_bg3072",
]

NGRAM_ORDER = [
    "record659_smoke",
    "lowrisk_smoke",
    "record659",
    "lowrisk",
    "lam10_conf05",
    "vr1_record659",
]

TTT_NGRAM_ORDER = [
    "record659_tttlr25_smoke",
    "lowrisk_tttlr25_smoke",
    "record659_tttlr25",
    "lowrisk_tttlr25",
    "vr1_record659_tttlr25",
]

ARTIFACT_LOG_NAMES = {
    "baseline": "h200_artifact_ttt_baseline.txt",
    "tttlr25": "h200_artifact_ttt_tttlr25.txt",
    "batch48": "h200_artifact_ttt_batch48.txt",
    "tttlr25_batch48": "h200_artifact_ttt_tttlr25_batch48.txt",
    "bg3072_tttlr25": "h200_artifact_ttt_bg3072_tttlr25.txt",
    "chunk16k": "h200_artifact_ttt_chunk16k.txt",
    "epochs2_tttlr25": "h200_artifact_ttt_epochs2_tttlr25.txt",
    "freeze2_tttlr25": "h200_artifact_ttt_freeze2_tttlr25.txt",
    "freeze2_epochs2_tttlr25": "h200_artifact_ttt_freeze2_epochs2_tttlr25.txt",
    "tttlr30": "h200_artifact_ttt_tttlr30.txt",
}

NGRAM_LOG_NAMES = {
    "record659": "h200_artifact_ngram_record659.txt",
    "record659_smoke": "h200_artifact_ngram_record659_smoke.txt",
    "lowrisk": "h200_artifact_ngram_lowrisk.txt",
    "lowrisk_smoke": "h200_artifact_ngram_lowrisk_smoke.txt",
    "lam10_conf05": "h200_artifact_ngram_lam10_conf05.txt",
    "vr1_record659": "h200_artifact_ngram_vr1_record659.txt",
}

TTT_NGRAM_LOG_NAMES = {
    "record659_tttlr25_smoke": "h200_artifact_ttt_ngram_record659_tttlr25_smoke.txt",
    "record659_tttlr25": "h200_artifact_ttt_ngram_record659_tttlr25.txt",
    "lowrisk_tttlr25_smoke": "h200_artifact_ttt_ngram_lowrisk_tttlr25_smoke.txt",
    "lowrisk_tttlr25": "h200_artifact_ttt_ngram_lowrisk_tttlr25.txt",
    "vr1_record659_tttlr25": "h200_artifact_ttt_ngram_vr1_record659_tttlr25.txt",
}

LEGAL_TTT_RE = re.compile(
    r"legal_ttt val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
LEGAL_TTT_NGRAM_RE = re.compile(
    r"legal_ttt_ngram val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
FINAL_NGRAM_RE = re.compile(
    r"final_ngram_eval val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+).*eval_time:(?P<eval_time_ms>\d+)ms"
)
SLIDING_RE = re.compile(
    r"final_int6_sliding_window val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+).*eval_time:(?P<eval_time_ms>\d+)ms"
)
STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+.*train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
STOP_RE = re.compile(r"stopping_early: wallclock_cap train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/")


def compose_slug(arch_candidate: str, ttt_candidate: str) -> str:
    if arch_candidate == "baseline":
        return ttt_candidate
    if ttt_candidate == "baseline":
        return arch_candidate
    if ttt_candidate.startswith("bg3072_") and "bg3072" in arch_candidate:
        return f"{arch_candidate}_{ttt_candidate.removeprefix('bg3072_')}"
    return f"{arch_candidate}_{ttt_candidate}"


def candidate_spec(source: str, candidate: str) -> dict[str, str]:
    if source == "artifact":
        return {"arch_candidate": "baseline", "ttt_candidate": candidate}
    if source == "proxy":
        return {"arch_candidate": candidate, "ttt_candidate": "baseline"}
    raise ValueError(f"unknown source: {source}")


def proxy_log_path(log_dir: Path, arch_candidate: str, ttt_candidate: str, seed: int) -> Path:
    slug = compose_slug(arch_candidate, ttt_candidate)
    if slug == "baseline":
        return log_dir / f"h200_ttt_h100proxy7185_seed{seed}.txt"
    return log_dir / f"h200_ttt_h100proxy7185_{slug}_seed{seed}.txt"


def parse_result(
    *,
    log_path: Path,
    label: str,
    source: str,
    arch_candidate: str,
    ttt_candidate: str,
    extra_log_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    result: dict[str, object] = {
        "label": label,
        "source": source,
        "arch_candidate": arch_candidate,
        "ttt_candidate": ttt_candidate,
        "log_path": str(log_path),
        "log_paths": [str(log_path), *[str(path) for path in extra_log_paths]],
        "completed": False,
        "submission_metric": None,
        "submission_val_bpb": None,
        "bytes_total": None,
        "submission_eval_time_ms": None,
        "legal_ttt_eval_time_ms": None,
        "legal_ttt_ngram_eval_time_ms": None,
        "final_ngram_eval_time_ms": None,
        "final_int6_sliding_window_exact": None,
        "final_int6_sliding_window_eval_time_ms": None,
        "last_step": None,
        "last_train_time_ms": None,
        "last_step_avg_ms": None,
    }
    existing_log_paths = [path for path in (log_path, *extra_log_paths) if path.exists()]
    if not existing_log_paths:
        return result

    summaries = [parse_log(path) for path in existing_log_paths]
    summary = merge_summaries(summaries) if len(summaries) > 1 else summaries[0]
    text = "\n".join(path.read_text(encoding="utf-8") for path in existing_log_paths)
    result["bytes_total"] = summary.get("bytes_total")
    result["submission_metric"] = summary.get("submission_metric")
    result["submission_val_bpb"] = summary.get("submission_val_bpb")
    result["last_step"] = summary.get("last_step")
    metric_eval_times: dict[str, int] = {}

    finals = summary.get("final_metrics", {})
    if isinstance(finals, dict):
        sliding = finals.get("final_int6_sliding_window_exact")
        if isinstance(sliding, dict):
            result["final_int6_sliding_window_exact"] = sliding.get("val_bpb")

    for line in text.splitlines():
        if match := LEGAL_TTT_RE.search(line):
            result["legal_ttt_eval_time_ms"] = int(match.group("eval_time_ms"))
            metric_eval_times["legal_ttt_exact"] = int(match.group("eval_time_ms"))
        if match := LEGAL_TTT_NGRAM_RE.search(line):
            result["legal_ttt_ngram_eval_time_ms"] = int(match.group("eval_time_ms"))
            metric_eval_times["legal_ttt_ngram_exact"] = int(match.group("eval_time_ms"))
        if match := FINAL_NGRAM_RE.search(line):
            result["final_ngram_eval_time_ms"] = int(match.group("eval_time_ms"))
            metric_eval_times["final_ngram_eval_exact"] = int(match.group("eval_time_ms"))
        if match := SLIDING_RE.search(line):
            result["final_int6_sliding_window_eval_time_ms"] = int(match.group("eval_time_ms"))
            metric_eval_times["final_int6_sliding_window_exact"] = int(match.group("eval_time_ms"))
        if match := STEP_RE.search(line):
            result["last_step"] = int(match.group("step"))
            result["last_train_time_ms"] = int(match.group("train_time_ms"))
            result["last_step_avg_ms"] = float(match.group("step_avg_ms"))
        if match := STOP_RE.search(line):
            result["last_step"] = int(match.group("step"))
            result["last_train_time_ms"] = int(match.group("train_time_ms"))

    submission_metric = result.get("submission_metric")
    if isinstance(submission_metric, str):
        result["submission_eval_time_ms"] = metric_eval_times.get(submission_metric)
    result["completed"] = result["submission_val_bpb"] is not None
    return result


def artifact_rank_key(result: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(result.get("submission_val_bpb") or math.inf),
        float(result.get("submission_eval_time_ms") or math.inf),
        float(result.get("bytes_total") or math.inf),
    )


def proxy_rank_key(result: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(result.get("submission_val_bpb") or math.inf),
        float(result.get("final_int6_sliding_window_exact") or math.inf),
        float(result.get("bytes_total") or math.inf),
    )


def promotion_rank_key(result: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(result.get("submission_val_bpb") or math.inf),
        float(result.get("bytes_total") or math.inf),
        float(result.get("submission_eval_time_ms") or math.inf),
        float(result.get("final_int6_sliding_window_exact") or math.inf),
    )


def choose_best(results: list[dict[str, object]], rank_key) -> dict[str, object] | None:
    completed = [result for result in results if result.get("completed")]
    if not completed:
        return None
    return sorted(completed, key=rank_key)[0]


def choose_best_nonbaseline(results: list[dict[str, object]], rank_key, source: str) -> dict[str, object] | None:
    completed = [result for result in results if result.get("completed")]
    if source == "artifact":
        filtered = [
            result
            for result in completed
            if result["ttt_candidate"] != "baseline" or result["arch_candidate"] != "baseline"
        ]
    else:
        filtered = [result for result in completed if result["arch_candidate"] != "baseline"]
    if not filtered:
        return None
    return sorted(filtered, key=rank_key)[0]


def h100_command(root_dir: Path, arch_candidate: str, ttt_candidate: str, seed: int = 1337) -> str:
    return (
        f"ARCH_CANDIDATE={arch_candidate} "
        f"TTT_CANDIDATE={ttt_candidate} "
        f"SEED={seed} "
        f"bash {root_dir / 'scripts/h100_record_push_candidate.sh'}"
    )


def h100_three_seed_command(root_dir: Path, arch_candidate: str, ttt_candidate: str) -> str:
    return (
        f"ARCH_CANDIDATE={arch_candidate} "
        f"TTT_CANDIDATE={ttt_candidate} "
        f'SEEDS="1337 42 2025" '
        f"bash {root_dir / 'scripts/h100_record_push_candidate_3seed.sh'}"
    )


def build_status(root_dir: Path, seed: int) -> dict[str, object]:
    record_dir = root_dir / RECORD_DIR_REL
    log_dir = record_dir / "logs"
    recovered_train_log = log_dir / "h200_ttt_recordstack_80shard_seed1337.txt"
    recovered_resume_log = log_dir / "h200_ttt_recordstack_80shard_seed1337_resume_ttt.txt"
    recovered_bytes_total = parse_log(recovered_train_log).get("bytes_total") if recovered_train_log.exists() else None

    artifact_results = []
    for candidate in ARTIFACT_ORDER:
        extra_log_paths: tuple[Path, ...] = ()
        log_path = log_dir / ARTIFACT_LOG_NAMES[candidate]
        if candidate == "baseline" and not log_path.exists():
            log_path = recovered_resume_log
            extra_log_paths = (recovered_train_log,)
        result = parse_result(
            log_path=log_path,
            label=candidate,
            source="artifact",
            extra_log_paths=extra_log_paths,
            **candidate_spec("artifact", candidate),
        )
        if result.get("bytes_total") is None and recovered_bytes_total is not None:
            result["bytes_total"] = recovered_bytes_total
        artifact_results.append(result)
    proxy_results = [
        parse_result(
            log_path=proxy_log_path(log_dir, candidate, "baseline", seed),
            label=candidate,
            source="proxy",
            **candidate_spec("proxy", candidate),
        )
        for candidate in PROXY_ORDER
    ]
    ngram_results = []
    for candidate in NGRAM_ORDER:
        result = parse_result(
            log_path=log_dir / NGRAM_LOG_NAMES[candidate],
            label=candidate,
            source="ngram",
            arch_candidate="baseline" if candidate != "vr1_record659" else "vr1",
            ttt_candidate=(
                "ngram659"
                if candidate in {"record659", "record659_smoke", "vr1_record659"}
                else "lowrisk_ngram"
                if candidate in {"lowrisk", "lowrisk_smoke"}
                else "lam10_conf05_ngram"
            ),
        )
        if result.get("bytes_total") is None and recovered_bytes_total is not None:
            result["bytes_total"] = recovered_bytes_total
        ngram_results.append(result)
    ttt_ngram_results = []
    for candidate in TTT_NGRAM_ORDER:
        result = parse_result(
            log_path=log_dir / TTT_NGRAM_LOG_NAMES[candidate],
            label=candidate,
            source="ttt_ngram",
            arch_candidate="baseline" if candidate != "vr1_record659_tttlr25" else "vr1",
            ttt_candidate=(
                "ngram659_tttlr25"
                if candidate in {"record659_tttlr25", "record659_tttlr25_smoke", "vr1_record659_tttlr25"}
                else "lowrisk_ngram_tttlr25"
            ),
        )
        if result.get("bytes_total") is None and recovered_bytes_total is not None:
            result["bytes_total"] = recovered_bytes_total
        ttt_ngram_results.append(result)

    artifact_best = choose_best(artifact_results, artifact_rank_key)
    proxy_best = choose_best(proxy_results, proxy_rank_key)
    ngram_best = choose_best(ngram_results, artifact_rank_key)
    ttt_ngram_best = choose_best(ttt_ngram_results, artifact_rank_key)
    artifact_nonbaseline = choose_best_nonbaseline(artifact_results, artifact_rank_key, "artifact")
    proxy_nonbaseline = choose_best_nonbaseline(proxy_results, proxy_rank_key, "proxy")

    recommended_combined: dict[str, object] | None = None
    combined_result: dict[str, object] | None = None
    if artifact_nonbaseline is not None and proxy_nonbaseline is not None:
        arch_candidate = str(proxy_nonbaseline["arch_candidate"])
        ttt_candidate = str(artifact_nonbaseline["ttt_candidate"])
        combined_slug = compose_slug(arch_candidate, ttt_candidate)
        combined_log_path = proxy_log_path(log_dir, arch_candidate, ttt_candidate, seed)
        combined_result = parse_result(
            log_path=combined_log_path,
            label=combined_slug,
            source="combined",
            arch_candidate=arch_candidate,
            ttt_candidate=ttt_candidate,
        )
        recommended_combined = {
            "arch_candidate": arch_candidate,
            "ttt_candidate": ttt_candidate,
            "label": combined_slug,
            "log_path": str(combined_log_path),
            "completed": bool(combined_result.get("completed")),
        }

    promotion_pool = [
        *[result for result in artifact_results if result.get("completed")],
        *[result for result in ngram_results if result.get("completed")],
        *[result for result in ttt_ngram_results if result.get("completed")],
        *[result for result in proxy_results if result.get("completed")],
    ]
    if combined_result is not None and combined_result.get("completed"):
        promotion_pool.append(combined_result)
    ranked_promotion_pool = sorted(promotion_pool, key=promotion_rank_key)
    promotion_pool = []
    seen_specs: set[tuple[str, str]] = set()
    for candidate in ranked_promotion_pool:
        spec = (str(candidate["arch_candidate"]), str(candidate["ttt_candidate"]))
        if spec in seen_specs:
            continue
        seen_specs.add(spec)
        promotion_pool.append(candidate)

    promoted = promotion_pool[0] if promotion_pool else None
    runner_up = promotion_pool[1] if len(promotion_pool) > 1 else None

    handoff = None
    if promoted is not None:
        handoff = {
            "acceptance_gate_bpb": ACCEPTANCE_GATE_BPB,
            "winner": promoted,
            "runner_up": runner_up,
            "winner_seed1337_command": h100_command(
                root_dir, str(promoted["arch_candidate"]), str(promoted["ttt_candidate"])
            ),
            "winner_three_seed_command": h100_three_seed_command(
                root_dir, str(promoted["arch_candidate"]), str(promoted["ttt_candidate"])
            ),
        }
        if runner_up is not None:
            handoff["runner_up_seed1337_command"] = h100_command(
                root_dir, str(runner_up["arch_candidate"]), str(runner_up["ttt_candidate"])
            )

    return {
        "record_dir": str(record_dir),
        "artifact_results": artifact_results,
        "artifact_ranked": sorted(
            [result for result in artifact_results if result.get("completed")],
            key=artifact_rank_key,
        ),
        "ngram_results": ngram_results,
        "ngram_ranked": sorted(
            [result for result in ngram_results if result.get("completed")],
            key=artifact_rank_key,
        ),
        "ttt_ngram_results": ttt_ngram_results,
        "ttt_ngram_ranked": sorted(
            [result for result in ttt_ngram_results if result.get("completed")],
            key=artifact_rank_key,
        ),
        "proxy_results": proxy_results,
        "proxy_ranked": sorted(
            [result for result in proxy_results if result.get("completed")],
            key=proxy_rank_key,
        ),
        "best_artifact": artifact_best,
        "best_ngram": ngram_best,
        "best_ttt_ngram": ttt_ngram_best,
        "best_artifact_nonbaseline": artifact_nonbaseline,
        "best_proxy": proxy_best,
        "best_proxy_nonbaseline": proxy_nonbaseline,
        "recommended_combined": recommended_combined,
        "combined_result": combined_result,
        "handoff": handoff,
    }


def print_ranked(title: str, results: list[dict[str, object]], *, secondary_key: str) -> None:
    print(title)
    if not results:
        print("  no completed logs yet")
        return
    for result in results:
        metric_name = result.get("submission_metric")
        print(
            "  "
            f"{result['label']}: {metric_name}={result.get('submission_val_bpb')} "
            f"{secondary_key}={result.get(secondary_key)} "
            f"bytes={result.get('bytes_total')} "
            f"log={result.get('log_path')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the H200-first record-push search state.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root. Defaults to the current checkout.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    status = build_status(args.root_dir.resolve(), args.seed)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    print_ranked(
        "Artifact-only TTT sweep",
        status["artifact_ranked"],
        secondary_key="legal_ttt_eval_time_ms",
    )
    print()
    print_ranked(
        "H200 H100-step proxy sweep",
        status["proxy_ranked"],
        secondary_key="final_int6_sliding_window_exact",
    )
    print()
    print_ranked(
        "Artifact-only n-gram sweep",
        status["ngram_ranked"],
        secondary_key="submission_eval_time_ms",
    )
    print()
    print_ranked(
        "Artifact-only TTT + n-gram sweep",
        status["ttt_ngram_ranked"],
        secondary_key="submission_eval_time_ms",
    )
    print()
    combined = status.get("recommended_combined")
    if combined is None:
        print("Combined proxy candidate")
        print("  not available yet: need a non-baseline artifact winner and a non-baseline proxy winner")
    else:
        print("Combined proxy candidate")
        print(
            "  "
            f"{combined['label']}: arch={combined['arch_candidate']} "
            f"ttt={combined['ttt_candidate']} "
            f"completed={combined['completed']} "
            f"log={combined['log_path']}"
        )
    print()
    print(f"Acceptance gate: legal_ttt_exact <= {ACCEPTANCE_GATE_BPB}")
    print(f"Record folder: {status['record_dir']}")
    print()
    handoff = status.get("handoff")
    if handoff is None:
        print("8xH100 handoff")
        print("  no completed candidates yet")
        return
    print("8xH100 handoff")
    winner = handoff["winner"]
    print(
        "  "
        f"winner={winner['label']} {winner.get('submission_metric')}={winner.get('submission_val_bpb')} "
        f"bytes={winner.get('bytes_total')}"
    )
    runner_up = handoff.get("runner_up")
    if runner_up is not None:
        print(
            "  "
            f"runner_up={runner_up['label']} {runner_up.get('submission_metric')}={runner_up.get('submission_val_bpb')} "
            f"bytes={runner_up.get('bytes_total')}"
        )
    print(f"  seed1337: {handoff['winner_seed1337_command']}")
    print(
        "  "
        f"if the exact 8xH100 seed clears bytes, train/eval time, and <= {ACCEPTANCE_GATE_BPB}: "
        f"{handoff['winner_three_seed_command']}"
    )
    if handoff.get("runner_up_seed1337_command"):
        print(f"  fallback_seed1337: {handoff['runner_up_seed1337_command']}")


if __name__ == "__main__":
    main()
