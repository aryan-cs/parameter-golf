#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from prepare_submission_metadata import merge_summaries, parse_log

# Public frontier claim as of 2026-03-25 from PR #674 ("Podracing: 1.0461 BPB").
CURRENT_PUBLIC_SOTA_BPB = 1.0461
RECORD_DELTA_NAT = 0.005
APPROX_BPB_PER_NAT = 0.5923
PRACTICAL_WIN_GATE_BPB = CURRENT_PUBLIC_SOTA_BPB - RECORD_DELTA_NAT * APPROX_BPB_PER_NAT
COMPETITION_ARTIFACT_LIMIT_BYTES = 16_000_000
COMPETITION_TRAIN_LIMIT_SECONDS = 600
COMPETITION_EVAL_LIMIT_SECONDS = 600
H100_PROXY_REFERENCE_STEPS = 7185
H100_PROXY_REFERENCE_STEP_AVG_MS = 83.4
H200_PROXY_STEP_AVG_MS = 765.97
H200_PROXY_TRAIN_LIMIT_MS = 5_503_469
H200_PROXY_TRAIN_LIMIT_SECONDS = H200_PROXY_TRAIN_LIMIT_MS / 1000.0
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
    "upstream_pr674_exact",
    "upstream_pr676_exact",
    "upstream_pr685_meanprob_exact",
    "upstream_pr685_phase1_exact",
    "upstream_pr684_exact",
    "podracing674",
    "podracing674_swiglu",
    "swiglu676",
    "swiglu",
    "xsa11",
    "podracing674_xsa11",
    "rope24",
    "vr1",
    "bg3072",
    "vr1_bg3072",
]

NGRAM_ORDER = [
    "record659_smoke",
    "record659",
    "record659_lamcool_smoke",
    "record659_lamcool",
    "record659_conf06_smoke",
    "record659_conf06",
    "record659_conf07_smoke",
    "record659_conf07",
    "record659_latecool_conf07_smoke",
    "record659_latecool_conf07",
    "record659_latecool_conf07_lamtail_smoke",
    "record659_latecool_conf07_lamtail",
    "record659_latecool_conf07_min4_smoke",
    "record659_latecool_conf07_min4",
    "record659_conf07_lamcool_smoke",
    "record659_conf07_lamcool",
    "record659_conf07_proxy7185",
    "record659_cool_conf07_smoke",
    "record659_cool_conf07",
    "record659_cool_conf07_lamcool_smoke",
    "record659_cool_conf07_lamcool",
    "record659_cool_conf07_min4_smoke",
    "record659_cool_conf07_min4",
    "record659_conf08_smoke",
    "record659_conf08",
    "record659_conf07_min4_smoke",
    "record659_conf07_min4",
    "record659_conf07_min5_smoke",
    "record659_conf07_min5",
    "record659_tgate30_smoke",
    "record659_tgate40_smoke",
    "record659_tgate40_min4_smoke",
    "record659_tgate40_min4",
    "record659_lam20_conf07_smoke",
    "record659_lam20_conf07",
    "record659_lam20_conf08_smoke",
    "record659_lam20_conf08",
    "record674_smoke",
    "record674",
    "record674_proxy7185",
    "record659_warm_conf07_smoke",
    "record659_warm_conf07",
    "record659_orderlam_smoke",
    "record659_orderlam",
    "record659_warm_conf07_orderlam_smoke",
    "record659_warm_conf07_orderlam",
    "record659_adapt_smoke",
    "record659_adapt",
    "record659_adapt_last2_smoke",
    "record659_adapt_last2",
    "record659_adapt_last4_smoke",
    "record659_adapt_last4",
    "lowrisk_smoke",
    "lowrisk",
    "lowrisk_adapt",
    "lam10_conf05",
    "vr1_record659",
]

TTT_NGRAM_ORDER = [
    "record659_tttlr25_smoke",
    "record659_late2_tttlr25_smoke",
    "record659_adamw5e4_late2_smoke",
    "record659_adamw1e4_late2_smoke",
    "record659_adamw30ep_cosine_smoke",
    "record659_adamw30ep_cosine_latecool_smoke",
    "record659_adamw30ep_cosine_lamcool_smoke",
    "record659_adamw30ep_cosine_lr3e4_smoke",
    "record659_adamw12ep_cosine_smoke",
    "lowrisk_tttlr25_smoke",
    "record659_late2_tttlr25",
    "record659_tttlr25",
    "record659_adamw1e4_late2",
    "record659_adamw5e4_late2",
    "record659_adamw30ep_cosine",
    "record659_adamw30ep_cosine_latecool",
    "record659_adamw30ep_cosine_lamcool",
    "record659_adamw30ep_cosine_lr3e4",
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
    "record659_lamcool": "h200_artifact_ngram_record659_lamcool.txt",
    "record659_lamcool_smoke": "h200_artifact_ngram_record659_lamcool_smoke.txt",
    "record659_conf06": "h200_artifact_ngram_record659_conf06.txt",
    "record659_conf06_smoke": "h200_artifact_ngram_record659_conf06_smoke.txt",
    "record659_conf07": "h200_artifact_ngram_record659_conf07.txt",
    "record659_conf07_smoke": "h200_artifact_ngram_record659_conf07_smoke.txt",
    "record659_latecool_conf07": "h200_artifact_ngram_record659_latecool_conf07.txt",
    "record659_latecool_conf07_smoke": "h200_artifact_ngram_record659_latecool_conf07_smoke.txt",
    "record659_latecool_conf07_lamtail": "h200_artifact_ngram_record659_latecool_conf07_lamtail.txt",
    "record659_latecool_conf07_lamtail_smoke": "h200_artifact_ngram_record659_latecool_conf07_lamtail_smoke.txt",
    "record659_latecool_conf07_min4": "h200_artifact_ngram_record659_latecool_conf07_min4.txt",
    "record659_latecool_conf07_min4_smoke": "h200_artifact_ngram_record659_latecool_conf07_min4_smoke.txt",
    "record659_conf07_lamcool": "h200_artifact_ngram_record659_conf07_lamcool.txt",
    "record659_conf07_lamcool_smoke": "h200_artifact_ngram_record659_conf07_lamcool_smoke.txt",
    "record659_conf07_proxy7185": "h200_artifact_ngram_record659_conf07_h100proxy7185_seed1337.txt",
    "record659_cool_conf07": "h200_artifact_ngram_record659_cool_conf07.txt",
    "record659_cool_conf07_smoke": "h200_artifact_ngram_record659_cool_conf07_smoke.txt",
    "record659_cool_conf07_lamcool": "h200_artifact_ngram_record659_cool_conf07_lamcool.txt",
    "record659_cool_conf07_lamcool_smoke": "h200_artifact_ngram_record659_cool_conf07_lamcool_smoke.txt",
    "record659_cool_conf07_min4": "h200_artifact_ngram_record659_cool_conf07_min4.txt",
    "record659_cool_conf07_min4_smoke": "h200_artifact_ngram_record659_cool_conf07_min4_smoke.txt",
    "record659_conf08": "h200_artifact_ngram_record659_conf08.txt",
    "record659_conf08_smoke": "h200_artifact_ngram_record659_conf08_smoke.txt",
    "record659_conf07_min4_smoke": "h200_artifact_ngram_record659_conf07_min4_smoke.txt",
    "record659_conf07_min4": "h200_artifact_ngram_record659_conf07_min4.txt",
    "record659_conf07_min5_smoke": "h200_artifact_ngram_record659_conf07_min5_smoke.txt",
    "record659_conf07_min5": "h200_artifact_ngram_record659_conf07_min5.txt",
    "record659_tgate30_smoke": "h200_artifact_ngram_record659_tgate30_smoke.txt",
    "record659_tgate40_smoke": "h200_artifact_ngram_record659_tgate40_smoke.txt",
    "record659_tgate40_min4_smoke": "h200_artifact_ngram_record659_tgate40_min4_smoke.txt",
    "record659_tgate40_min4": "h200_artifact_ngram_record659_tgate40_min4.txt",
    "record659_lam20_conf07_smoke": "h200_artifact_ngram_record659_lam20_conf07_smoke.txt",
    "record659_lam20_conf07": "h200_artifact_ngram_record659_lam20_conf07.txt",
    "record659_lam20_conf08_smoke": "h200_artifact_ngram_record659_lam20_conf08_smoke.txt",
    "record659_lam20_conf08": "h200_artifact_ngram_record659_lam20_conf08.txt",
    "record674_smoke": "h200_artifact_ngram_record674_smoke.txt",
    "record674": "h200_artifact_ngram_record674.txt",
    "record674_proxy7185": "h200_artifact_ngram_record674_h100proxy7185_seed1337.txt",
    "record659_warm_conf07": "h200_artifact_ngram_record659_warm_conf07.txt",
    "record659_warm_conf07_smoke": "h200_artifact_ngram_record659_warm_conf07_smoke.txt",
    "record659_orderlam": "h200_artifact_ngram_record659_orderlam.txt",
    "record659_orderlam_smoke": "h200_artifact_ngram_record659_orderlam_smoke.txt",
    "record659_warm_conf07_orderlam": "h200_artifact_ngram_record659_warm_conf07_orderlam.txt",
    "record659_warm_conf07_orderlam_smoke": "h200_artifact_ngram_record659_warm_conf07_orderlam_smoke.txt",
    "record659_adapt_smoke": "h200_artifact_ngram_record659_adapt_smoke.txt",
    "record659_adapt": "h200_artifact_ngram_record659_adapt.txt",
    "record659_adapt_last2_smoke": "h200_artifact_ngram_record659_adapt_last2_smoke.txt",
    "record659_adapt_last2": "h200_artifact_ngram_record659_adapt_last2.txt",
    "record659_adapt_last4_smoke": "h200_artifact_ngram_record659_adapt_last4_smoke.txt",
    "record659_adapt_last4": "h200_artifact_ngram_record659_adapt_last4.txt",
    "lowrisk": "h200_artifact_ngram_lowrisk.txt",
    "lowrisk_smoke": "h200_artifact_ngram_lowrisk_smoke.txt",
    "lowrisk_adapt": "h200_artifact_ngram_lowrisk_adapt.txt",
    "lam10_conf05": "h200_artifact_ngram_lam10_conf05.txt",
    "vr1_record659": "h200_artifact_ngram_vr1_record659.txt",
}

TTT_NGRAM_LOG_NAMES = {
    "record659_tttlr25_smoke": "h200_artifact_ttt_ngram_record659_tttlr25_smoke.txt",
    "record659_late2_tttlr25_smoke": "h200_artifact_ttt_ngram_record659_late2_tttlr25_smoke.txt",
    "record659_adamw5e4_late2_smoke": "h200_artifact_ttt_ngram_record659_adamw5e4_late2_smoke.txt",
    "record659_adamw1e4_late2_smoke": "h200_artifact_ttt_ngram_record659_adamw1e4_late2_smoke.txt",
    "record659_adamw30ep_cosine_smoke": "h200_artifact_ttt_ngram_record659_adamw30ep_cosine_smoke.txt",
    "record659_adamw30ep_cosine_latecool_smoke": "h200_artifact_ttt_ngram_record659_adamw30ep_cosine_latecool_smoke.txt",
    "record659_adamw30ep_cosine_lamcool_smoke": "h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lamcool_smoke.txt",
    "record659_adamw30ep_cosine_lr3e4_smoke": "h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lr3e4_smoke.txt",
    "record659_adamw12ep_cosine_smoke": "h200_artifact_ttt_ngram_record659_adamw12ep_cosine_smoke.txt",
    "record659_late2_tttlr25": "h200_artifact_ttt_ngram_record659_late2_tttlr25.txt",
    "record659_tttlr25": "h200_artifact_ttt_ngram_record659_tttlr25.txt",
    "record659_adamw1e4_late2": "h200_artifact_ttt_ngram_record659_adamw1e4_late2.txt",
    "record659_adamw5e4_late2": "h200_artifact_ttt_ngram_record659_adamw5e4_late2.txt",
    "record659_adamw30ep_cosine": "h200_artifact_ttt_ngram_record659_adamw30ep_cosine.txt",
    "record659_adamw30ep_cosine_latecool": "h200_artifact_ttt_ngram_record659_adamw30ep_cosine_latecool.txt",
    "record659_adamw30ep_cosine_lamcool": "h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lamcool.txt",
    "record659_adamw30ep_cosine_lr3e4": "h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lr3e4.txt",
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
FINAL_SLIDING_NGRAM_RE = re.compile(
    r"final_int6_sliding_window_ngram(?P<order>\d+) val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
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
    if arch_candidate == "upstream_pr674_exact":
        return log_dir / f"h200_upstream_pr674_proxy7185_seed{seed}.txt"
    if arch_candidate == "upstream_pr676_exact":
        return log_dir / f"h200_upstream_pr676_proxy7185_seed{seed}.txt"
    if arch_candidate == "upstream_pr685_meanprob_exact":
        return log_dir / f"h200_upstream_pr685_meanprob_proxy7185_seed{seed}.txt"
    if arch_candidate == "upstream_pr685_phase1_exact":
        return log_dir / f"h200_upstream_pr685_phase1_proxy7185_seed{seed}.txt"
    if arch_candidate == "upstream_pr684_exact":
        return log_dir / f"h200_upstream_pr684_proxy6555_seed{seed}.txt"
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
        "final_sliding_ngram_eval_time_ms": None,
        "final_int6_sliding_window_exact": None,
        "final_int6_sliding_window_eval_time_ms": None,
        "last_step": None,
        "last_train_time_ms": None,
        "last_step_avg_ms": None,
        "artifact_cap_pass": None,
        "h200_dev_train_cap_pass": None,
    }
    if not log_path.exists():
        return result
    existing_log_paths = [log_path, *[path for path in extra_log_paths if path.exists()]]

    summaries = [parse_log(path) for path in existing_log_paths]
    current_summary = summaries[0]
    summary = merge_summaries(summaries) if len(summaries) > 1 else current_summary
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
        if match := FINAL_SLIDING_NGRAM_RE.search(line):
            metric_name = f"final_int6_sliding_window_ngram{match.group('order')}_exact"
            result["final_sliding_ngram_eval_time_ms"] = int(match.group("eval_time_ms"))
            metric_eval_times[metric_name] = int(match.group("eval_time_ms"))
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

    if source == "ngram":
        submission_metric = current_summary.get("submission_metric")
        if isinstance(submission_metric, str) and (
            submission_metric == "final_ngram_eval_exact"
            or (submission_metric.startswith("final_int6_sliding_window_ngram") and submission_metric.endswith("_exact"))
        ):
            result["submission_metric"] = submission_metric
            result["submission_val_bpb"] = current_summary.get("submission_val_bpb")
        else:
            result["submission_metric"] = None
            result["submission_val_bpb"] = None
    elif source == "ttt_ngram":
        if current_summary.get("submission_metric") == "legal_ttt_ngram_exact":
            result["submission_metric"] = "legal_ttt_ngram_exact"
            result["submission_val_bpb"] = current_summary.get("submission_val_bpb")
        else:
            result["submission_metric"] = None
            result["submission_val_bpb"] = None

    submission_metric = result.get("submission_metric")
    if isinstance(submission_metric, str):
        result["submission_eval_time_ms"] = metric_eval_times.get(submission_metric)
    bytes_total = result.get("bytes_total")
    if isinstance(bytes_total, int):
        result["artifact_cap_pass"] = bytes_total <= COMPETITION_ARTIFACT_LIMIT_BYTES
    train_time_ms = result.get("last_train_time_ms")
    last_step = result.get("last_step")
    if isinstance(train_time_ms, int) and isinstance(last_step, int):
        result["h200_dev_train_cap_pass"] = (
            train_time_ms <= H200_PROXY_TRAIN_LIMIT_MS and last_step <= H100_PROXY_REFERENCE_STEPS
        )
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
    if arch_candidate == "upstream_pr674_exact":
        return f"SEED={seed} bash {root_dir / 'scripts/h100_upstream_pr674_exact.sh'}"
    if arch_candidate == "upstream_pr676_exact":
        return f"SEED={seed} bash {root_dir / 'scripts/h100_upstream_pr676_exact.sh'}"
    if arch_candidate == "upstream_pr685_meanprob_exact":
        return f"SEED={seed} bash {root_dir / 'scripts/h100_upstream_pr685_meanprob_exact.sh'}"
    if arch_candidate == "upstream_pr685_phase1_exact":
        return f"SEED={seed} bash {root_dir / 'scripts/h100_upstream_pr685_phase1_exact.sh'}"
    if arch_candidate == "upstream_pr684_exact":
        return f"SEED={seed} bash {root_dir / 'scripts/h100_upstream_pr684_exact.sh'}"
    return (
        f"ARCH_CANDIDATE={arch_candidate} "
        f"TTT_CANDIDATE={ttt_candidate} "
        f"SEED={seed} "
        f"bash {root_dir / 'scripts/h100_record_push_candidate.sh'}"
    )


def h100_three_seed_command(root_dir: Path, arch_candidate: str, ttt_candidate: str) -> str:
    if arch_candidate == "upstream_pr674_exact":
        return f"bash {root_dir / 'scripts/h100_upstream_pr674_exact_3seed.sh'}"
    if arch_candidate == "upstream_pr676_exact":
        return f"bash {root_dir / 'scripts/h100_upstream_pr676_exact_3seed.sh'}"
    if arch_candidate == "upstream_pr685_meanprob_exact":
        return f"bash {root_dir / 'scripts/h100_upstream_pr685_meanprob_exact_3seed.sh'}"
    if arch_candidate == "upstream_pr685_phase1_exact":
        return f"bash {root_dir / 'scripts/h100_upstream_pr685_phase1_exact_3seed.sh'}"
    if arch_candidate == "upstream_pr684_exact":
        return f"bash {root_dir / 'scripts/h100_upstream_pr684_exact_3seed.sh'}"
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
        extra_log_paths: tuple[Path, ...] = (recovered_train_log,)
        log_path = log_dir / ARTIFACT_LOG_NAMES[candidate]
        if candidate == "baseline" and not log_path.exists():
            log_path = recovered_resume_log
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
        extra_log_paths = (
            (proxy_log_path(log_dir, "baseline", "baseline", seed),)
            if candidate in {"record659_conf07_proxy7185", "record674_proxy7185"}
            else (recovered_train_log,)
        )
        result = parse_result(
            log_path=log_dir / NGRAM_LOG_NAMES[candidate],
            label=candidate,
            source="ngram",
            extra_log_paths=extra_log_paths,
            arch_candidate="baseline" if candidate != "vr1_record659" else "vr1",
            ttt_candidate=(
                "ngram659_conf06"
                if candidate in {"record659_conf06", "record659_conf06_smoke"}
                else "ngram659_latecool_conf07_lamtail"
                if candidate in {"record659_latecool_conf07_lamtail", "record659_latecool_conf07_lamtail_smoke"}
                else "ngram659_latecool_conf07_min4"
                if candidate in {"record659_latecool_conf07_min4", "record659_latecool_conf07_min4_smoke"}
                else "ngram659_latecool_conf07"
                if candidate in {"record659_latecool_conf07", "record659_latecool_conf07_smoke"}
                else "ngram659_conf07_lamcool"
                if candidate in {"record659_conf07_lamcool", "record659_conf07_lamcool_smoke"}
                else "ngram659_cool_conf07_lamcool"
                if candidate in {"record659_cool_conf07_lamcool", "record659_cool_conf07_lamcool_smoke"}
                else "ngram659_cool_conf07_min4"
                if candidate in {"record659_cool_conf07_min4", "record659_cool_conf07_min4_smoke"}
                else "ngram659_cool_conf07"
                if candidate in {"record659_cool_conf07", "record659_cool_conf07_smoke"}
                else "ngram659_conf08"
                if candidate in {"record659_conf08", "record659_conf08_smoke"}
                else "ngram659_conf07_min4"
                if candidate in {"record659_conf07_min4", "record659_conf07_min4_smoke"}
                else "ngram659_conf07_min5"
                if candidate in {"record659_conf07_min5", "record659_conf07_min5_smoke"}
                else "ngram659_conf07_lam20"
                if candidate in {"record659_lam20_conf07", "record659_lam20_conf07_smoke"}
                else "ngram659_tgate40_min4"
                if candidate in {"record659_tgate40_min4", "record659_tgate40_min4_smoke"}
                else "ngram659_lamcool"
                if candidate in {"record659_lamcool", "record659_lamcool_smoke"}
                else "ngram659_conf07"
                if candidate in {"record659_conf07", "record659_conf07_smoke", "record659_conf07_proxy7185"}
                else "ngram674"
                if candidate in {"record674", "record674_smoke", "record674_proxy7185"}
                else
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
            extra_log_paths=(recovered_train_log,),
            arch_candidate="baseline" if candidate != "vr1_record659_tttlr25" else "vr1",
            ttt_candidate=(
                "ngram659_adamw30ep_cosine_lr3e4"
                if candidate in {"record659_adamw30ep_cosine_lr3e4", "record659_adamw30ep_cosine_lr3e4_smoke"}
                else "ngram659_adamw30ep_cosine_latecool"
                if candidate in {"record659_adamw30ep_cosine_latecool", "record659_adamw30ep_cosine_latecool_smoke"}
                else "ngram659_adamw30ep_cosine_lamcool"
                if candidate in {"record659_adamw30ep_cosine_lamcool", "record659_adamw30ep_cosine_lamcool_smoke"}
                else "ngram659_adamw30ep_cosine"
                if candidate in {"record659_adamw30ep_cosine", "record659_adamw30ep_cosine_smoke"}
                else "ngram659_adamw12ep_cosine"
                if candidate in {"record659_adamw12ep_cosine", "record659_adamw12ep_cosine_smoke"}
                else
                "ngram659_late2_adamw5e4"
                if candidate in {"record659_adamw5e4_late2", "record659_adamw5e4_late2_smoke"}
                else "ngram659_late2_adamw1e4"
                if candidate in {"record659_adamw1e4_late2", "record659_adamw1e4_late2_smoke"}
                else "ngram659_late2_tttlr25"
                if candidate in {"record659_late2_tttlr25", "record659_late2_tttlr25_smoke"}
                else
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
        *[result for result in artifact_results if result.get("completed") and not str(result["label"]).endswith("_smoke")],
        *[result for result in ngram_results if result.get("completed") and not str(result["label"]).endswith("_smoke")],
        *[result for result in ttt_ngram_results if result.get("completed") and not str(result["label"]).endswith("_smoke")],
        *[result for result in proxy_results if result.get("completed") and not str(result["label"]).endswith("_smoke")],
    ]
    if combined_result is not None and combined_result.get("completed"):
        promotion_pool.append(combined_result)
    heuristic_valid_pool = [
        candidate
        for candidate in promotion_pool
        if candidate.get("artifact_cap_pass") is not False and candidate.get("h200_dev_train_cap_pass") is not False
    ]
    ranked_promotion_pool = sorted(heuristic_valid_pool, key=promotion_rank_key)
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
            "current_public_sota_bpb": CURRENT_PUBLIC_SOTA_BPB,
            "practical_win_gate_bpb": PRACTICAL_WIN_GATE_BPB,
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
        extras: list[str] = []
        if result.get("last_train_time_ms") is not None:
            extras.append(f"train_ms={result.get('last_train_time_ms')}")
        if result.get("h200_dev_train_cap_pass") is not None:
            extras.append(
                "h200_dev_train_cap="
                + ("ok" if bool(result.get("h200_dev_train_cap_pass")) else "fail")
            )
        if result.get("artifact_cap_pass") is not None:
            extras.append(
                "artifact_cap="
                + ("ok" if bool(result.get("artifact_cap_pass")) else "fail")
            )
        extras_text = (" ".join(extras) + " ") if extras else ""
        print(
            "  "
            f"{result['label']}: {metric_name}={result.get('submission_val_bpb')} "
            f"{secondary_key}={result.get(secondary_key)} "
            f"bytes={result.get('bytes_total')} "
            f"{extras_text}"
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
    print(f"Current public SOTA (2026-03-25): {CURRENT_PUBLIC_SOTA_BPB:.4f}")
    print(f"Approx record-claim gate (0.005 nat better): <= {PRACTICAL_WIN_GATE_BPB:.4f}")
    print(f"Record folder: {status['record_dir']}")
    print("Constraint guardrails")
    print(f"  competition_artifact_cap_bytes={COMPETITION_ARTIFACT_LIMIT_BYTES}")
    print(f"  competition_train_cap_seconds={COMPETITION_TRAIN_LIMIT_SECONDS}")
    print(f"  competition_eval_cap_seconds={COMPETITION_EVAL_LIMIT_SECONDS}")
    print(
        "  "
        f"h200_dev_train_proxy_cap={H200_PROXY_TRAIN_LIMIT_MS}ms "
        f"(~{H200_PROXY_TRAIN_LIMIT_SECONDS / 60.0:.1f} min) "
        f"at <= {H100_PROXY_REFERENCE_STEPS} steps"
    )
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
        f"if the exact 8xH100 seed clears bytes, train/eval time, and <= {handoff['practical_win_gate_bpb']:.4f}: "
        f"{handoff['winner_three_seed_command']}"
    )
    if handoff.get("runner_up_seed1337_command"):
        print(f"  fallback_seed1337: {handoff['runner_up_seed1337_command']}")


if __name__ == "__main__":
    main()
