#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "loop" / "config.json"
LEADERBOARD_URL = "https://raw.githubusercontent.com/openai/parameter-golf/main/README.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size > 0 and not content.startswith("\n"):
            f.write("\n")
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")


def append_journal_entry(cfg: "Config", title: str, lines: list[str]) -> None:
    timestamp = utc_now()
    filtered = [line for line in lines if line]
    body = "\n".join(f"- {line}" for line in filtered)
    append_markdown(cfg.journal_file, f"## {timestamp} - {title}\n\n{body}\n")


@dataclass
class Config:
    workdir: Path
    runtime_dir: Path
    manifest_inbox_dirs: list[Path]
    goal_file: Path
    prompt_template_file: Path
    agent_output_schema_file: Path
    journal_file: Path
    repo_context_file: Path | None
    codex_model: str
    codex_args: list[str]
    codex_timeout_seconds: int
    default_sleep_seconds: int
    leaderboard_refresh_seconds: int
    max_concurrent_jobs: int
    default_job_timeout_seconds: int
    allow_agent_halt: bool
    record_only: bool
    desired_rank: int
    stop_when_best_bpb_le: float | None
    auto_git_push: bool
    git_remote_name: str
    auto_git_commit_prefix: str


def load_config(path: Path) -> Config:
    raw = load_json(path)

    def resolve(p: str) -> Path:
        candidate = Path(p)
        if candidate.is_absolute():
            return candidate
        return (ROOT / candidate).resolve()

    def resolve_optional(p: Any) -> Path | None:
        if p is None:
            return None
        return resolve(str(p))

    stop_bpb = raw.get("stop_when_best_bpb_le")
    stop_bpb = float(stop_bpb) if stop_bpb is not None else None
    return Config(
        workdir=resolve(raw.get("workdir", str(ROOT))),
        runtime_dir=resolve(raw.get("runtime_dir", "loop/runtime")),
        manifest_inbox_dirs=[resolve(str(p)) for p in raw.get("manifest_inbox_dirs", ["../research-experiments/manifests/pending"])],
        goal_file=resolve(raw.get("goal_file", "loop/goal.md")),
        prompt_template_file=resolve(raw.get("prompt_template_file", "loop/prompt_template.md")),
        agent_output_schema_file=resolve(raw.get("agent_output_schema_file", "loop/agent_output.schema.json")),
        journal_file=resolve(raw.get("journal_file", "JOURNAL.md")),
        repo_context_file=resolve_optional(raw.get("repo_context_file")),
        codex_model=str(raw.get("codex_model", "gpt-5.4")),
        codex_args=list(raw.get("codex_args", ["--full-auto", "--ephemeral", "--skip-git-repo-check", "--color", "never"])),
        codex_timeout_seconds=int(raw.get("codex_timeout_seconds", 1800)),
        default_sleep_seconds=int(raw.get("default_sleep_seconds", 60)),
        leaderboard_refresh_seconds=int(raw.get("leaderboard_refresh_seconds", 1800)),
        max_concurrent_jobs=int(raw.get("max_concurrent_jobs", 1)),
        default_job_timeout_seconds=int(raw.get("default_job_timeout_seconds", 28800)),
        allow_agent_halt=bool(raw.get("allow_agent_halt", False)),
        record_only=bool(raw.get("record_only", True)),
        desired_rank=int(raw.get("desired_rank", 3)),
        stop_when_best_bpb_le=stop_bpb,
        auto_git_push=bool(raw.get("auto_git_push", True)),
        git_remote_name=str(raw.get("git_remote_name", "origin")),
        auto_git_commit_prefix=str(raw.get("auto_git_commit_prefix", "loop checkpoint")),
    )


def ensure_runtime_layout(runtime_dir: Path) -> None:
    for rel in (
        "logs",
        "reports",
        "jobs",
        "queue/pending",
        "queue/running",
        "queue/done",
        "queue/failed",
        "queue/blocked",
    ):
        (runtime_dir / rel).mkdir(parents=True, exist_ok=True)


def open_db(runtime_dir: Path) -> sqlite3.Connection:
    db_path = runtime_dir / "state.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS controller_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
            fetched_at TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            top1_run TEXT,
            top1_bpb REAL,
            top2_run TEXT,
            top2_bpb REAL,
            top3_run TEXT,
            top3_bpb REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS codex_turns (
            turn_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            exit_code INTEGER NOT NULL,
            prompt_path TEXT NOT NULL,
            output_path TEXT,
            log_path TEXT NOT NULL,
            summary TEXT,
            decision TEXT,
            sleep_seconds INTEGER,
            best_score_bpb REAL,
            needs_human_attention INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            job_kind TEXT NOT NULL DEFAULT 'task',
            description TEXT,
            status TEXT NOT NULL,
            command TEXT NOT NULL,
            cwd TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            log_path TEXT NOT NULL,
            script_path TEXT NOT NULL,
            exit_file TEXT NOT NULL,
            pid INTEGER,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            timeout_seconds INTEGER NOT NULL,
            exit_code INTEGER,
            val_bpb REAL,
            artifact_bytes INTEGER,
            train_time_ms INTEGER,
            notes TEXT
        )
        """
    )
    ensure_table_column(conn, "jobs", "job_kind", "TEXT NOT NULL DEFAULT 'task'")
    conn.commit()
    return conn


def ensure_table_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    conn.commit()


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM controller_state WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO controller_state(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


def delete_state(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("DELETE FROM controller_state WHERE key = ?", (key,))
    conn.commit()


def get_active_codex_turn(conn: sqlite3.Connection) -> dict[str, Any] | None:
    raw = get_state(conn, "active_codex_turn")
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def set_active_codex_turn(
    conn: sqlite3.Connection,
    *,
    turn_id: str,
    started_at: str,
    prompt_path: Path,
    output_path: Path,
    log_path: Path,
    pid: int | None,
) -> None:
    set_state(
        conn,
        "active_codex_turn",
        json.dumps(
            {
                "turn_id": turn_id,
                "started_at": started_at,
                "prompt_path": str(prompt_path),
                "output_path": str(output_path),
                "log_path": str(log_path),
                "pid": pid,
            },
            sort_keys=True,
        ),
    )


def clear_active_codex_turn(conn: sqlite3.Connection) -> None:
    delete_state(conn, "active_codex_turn")


def active_codex_turn_alive(active_turn: dict[str, Any] | None) -> bool:
    if active_turn is None:
        return False
    pid = active_turn.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def clear_stale_active_codex_turn(conn: sqlite3.Connection) -> bool:
    active_turn = get_active_codex_turn(conn)
    if active_turn is None or active_codex_turn_alive(active_turn):
        return False
    clear_active_codex_turn(conn)
    return True


def git_repo_root(start: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def git_status_porcelain(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def maybe_checkpoint_repo(cfg: Config, reason: str) -> bool:
    if not cfg.auto_git_push:
        return False
    repo_root = git_repo_root(cfg.workdir)
    if repo_root is None:
        print("[loop] git checkpoint skipped: no git repo root found")
        return False

    status_lines = git_status_porcelain(repo_root)
    if not status_lines:
        return False

    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=False)
    commit_message = f"{cfg.auto_git_commit_prefix}: {reason} @ {utc_now()}"
    commit = subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if commit.returncode != 0:
        if "nothing to commit" in (commit.stdout + commit.stderr).lower():
            return False
        print("[loop] git commit failed")
        if commit.stdout.strip():
            print(commit.stdout.strip())
        if commit.stderr.strip():
            print(commit.stderr.strip())
        return False

    push = subprocess.run(
        ["git", "push", cfg.git_remote_name, "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if push.returncode != 0:
        print("[loop] git push failed")
        if push.stdout.strip():
            print(push.stdout.strip())
        if push.stderr.strip():
            print(push.stderr.strip())
        return False

    print(f"[loop] git checkpoint pushed: {commit_message}")
    return True


def ensure_manifest_inboxes(cfg: Config) -> None:
    for inbox in cfg.manifest_inbox_dirs:
        inbox.mkdir(parents=True, exist_ok=True)


def staged_manifest_count(cfg: Config) -> int:
    return sum(len(list(inbox.glob("*.json"))) for inbox in cfg.manifest_inbox_dirs)


def manifest_known_to_runtime(cfg: Config, name: str) -> bool:
    for status in ("pending", "running", "done", "failed", "blocked"):
        if (cfg.runtime_dir / "queue" / status / name).exists():
            return True
    return False


def manifest_preflight_issues(cfg: Config, manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for module_name in manifest.get("required_python_modules", []):
        if importlib.util.find_spec(str(module_name)) is None:
            issues.append(f"missing python module: {module_name}")

    required_paths = manifest.get("required_paths", [])
    if isinstance(required_paths, list):
        for raw_path in required_paths:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = (cfg.workdir / path).resolve()
            if not path.exists():
                issues.append(f"missing path: {path}")

    required_cuda_devices = manifest.get("required_cuda_devices")
    if required_cuda_devices is not None:
        try:
            import torch  # type: ignore
        except Exception:
            issues.append("missing python module: torch")
        else:
            if not torch.cuda.is_available():
                issues.append("CUDA unavailable")
            elif torch.cuda.device_count() < int(required_cuda_devices):
                issues.append(
                    f"need {int(required_cuda_devices)} CUDA devices, found {torch.cuda.device_count()}"
                )
    deduped: list[str] = []
    for issue in issues:
        if issue not in deduped:
            deduped.append(issue)
    return deduped


def ingest_manifest_inboxes(cfg: Config, conn: sqlite3.Connection) -> int:
    pending_dir = cfg.runtime_dir / "queue" / "pending"
    imported = 0
    for inbox in cfg.manifest_inbox_dirs:
        for manifest_path in sorted(inbox.glob("*.json")):
            if manifest_known_to_runtime(cfg, manifest_path.name):
                continue
            try:
                manifest = load_json(manifest_path)
            except Exception as exc:
                key = f"manifest_error::{manifest_path}"
                msg = f"invalid manifest JSON: {exc}"
                if get_state(conn, key) != msg:
                    append_journal_entry(
                        cfg,
                        f"Manifest Invalid: {manifest_path.stem}",
                        [f"Path: {manifest_path}", msg],
                    )
                    set_state(conn, key, msg)
                continue

            issues = manifest_preflight_issues(cfg, manifest if isinstance(manifest, dict) else {})
            if issues:
                key = f"manifest_wait::{manifest_path}"
                issue_text = " | ".join(issues)
                if get_state(conn, key) != issue_text:
                    append_journal_entry(
                        cfg,
                        f"Manifest Waiting For Runtime: {manifest_path.stem}",
                        [f"Path: {manifest_path}", *issues],
                    )
                    set_state(conn, key, issue_text)
                continue

            shutil.move(str(manifest_path), pending_dir / manifest_path.name)
            imported += 1
            append_journal_entry(
                cfg,
                f"Manifest Ingested: {manifest_path.stem}",
                [
                    f"From: {manifest_path}",
                    f"To: {pending_dir / manifest_path.name}",
                ],
            )
    return imported


def fetch_leaderboard_markdown() -> str:
    with urllib.request.urlopen(LEADERBOARD_URL, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_leaderboard(markdown: str) -> dict[str, Any]:
    lines = markdown.splitlines()
    in_table = False
    rows: list[tuple[str, float]] = []
    for line in lines:
        if line.strip() == "## Leaderboard":
            in_table = True
            continue
        if in_table and line.startswith("#### "):
            break
        if not in_table:
            continue
        if not line.startswith("|"):
            continue
        if line.startswith("| Run |") or line.startswith("|-----"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 2:
            continue
        run = parts[0]
        score_text = parts[1]
        try:
            score = float(score_text)
        except ValueError:
            continue
        rows.append((run, score))
        if len(rows) >= 3:
            break
    if len(rows) < 3:
        raise ValueError("Failed to parse top 3 leaderboard rows")
    return {
        "top1_run": rows[0][0],
        "top1_bpb": rows[0][1],
        "top2_run": rows[1][0],
        "top2_bpb": rows[1][1],
        "top3_run": rows[2][0],
        "top3_bpb": rows[2][1],
    }


def refresh_leaderboard_if_needed(cfg: Config, conn: sqlite3.Connection) -> dict[str, Any] | None:
    last_fetch = get_state(conn, "last_leaderboard_fetch")
    now = time.time()
    if last_fetch is not None and now - float(last_fetch) < cfg.leaderboard_refresh_seconds:
        row = conn.execute(
            "SELECT * FROM leaderboard_snapshots ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None
    try:
        md = fetch_leaderboard_markdown()
        parsed = parse_leaderboard(md)
    except Exception as exc:
        print(f"[loop] leaderboard refresh failed: {exc}", file=sys.stderr)
        row = conn.execute(
            "SELECT * FROM leaderboard_snapshots ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None
    fetched_at = utc_now()
    conn.execute(
        """
        INSERT INTO leaderboard_snapshots(
            fetched_at, source_url, top1_run, top1_bpb, top2_run, top2_bpb, top3_run, top3_bpb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fetched_at,
            LEADERBOARD_URL,
            parsed["top1_run"],
            parsed["top1_bpb"],
            parsed["top2_run"],
            parsed["top2_bpb"],
            parsed["top3_run"],
            parsed["top3_bpb"],
        ),
    )
    conn.commit()
    set_state(conn, "last_leaderboard_fetch", str(now))
    row = conn.execute(
        "SELECT * FROM leaderboard_snapshots ORDER BY fetched_at DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row is not None else None


def summarize_leaderboard(row: dict[str, Any] | None, cfg: Config) -> str:
    if row is None:
        return "No leaderboard snapshot available."
    target_note = (
        "Record-only mode is enabled, so beating #3 alone is not sufficient for a final stop."
        if cfg.record_only
        else f"Desired rank target is top {cfg.desired_rank}."
    )
    return textwrap.dedent(
        f"""\
        Source: {row['source_url']}
        Snapshot time: {row['fetched_at']}
        #1: {row['top1_run']} ({row['top1_bpb']:.4f})
        #2: {row['top2_run']} ({row['top2_bpb']:.4f})
        #3: {row['top3_run']} ({row['top3_bpb']:.4f})
        {target_note}
        """
    ).strip()


def parse_metrics_from_log(text: str) -> dict[str, Any]:
    def last_float(patterns: list[str]) -> float | None:
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.MULTILINE)
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        return None

    def last_int(patterns: list[str]) -> int | None:
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.MULTILINE)
            if matches:
                try:
                    return int(matches[-1])
                except ValueError:
                    continue
        return None

    return {
        "val_bpb": last_float(
            [
                r"final_[^\n]*val_bpb[:=]([0-9]+\.[0-9]+)",
                r"\bval_bpb[:=]([0-9]+\.[0-9]+)",
            ]
        ),
        "artifact_bytes": last_int(
            [
                r"Total submission size[^:]*:\s*([0-9]+)\s*bytes",
                r"total_artifact_bytes[:=]([0-9]+)",
                r"artifact_bytes[^0-9]*([0-9]+)",
            ]
        ),
        "train_time_ms": last_int([r"train_time[:=]([0-9]+)ms", r"train_time_ms[:=]([0-9]+)"]),
    }


def parse_metrics_from_stats(payload: dict[str, Any]) -> dict[str, Any]:
    def first_present(*keys: str) -> Any:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        return None

    val_bpb = first_present("final_val_bpb", "best_val_bpb", "val_bpb")
    artifact_bytes = first_present("total_artifact_bytes", "artifact_bytes")
    train_time_ms = first_present("train_time_ms")

    return {
        "val_bpb": None if val_bpb is None else float(val_bpb),
        "artifact_bytes": None if artifact_bytes is None else int(artifact_bytes),
        "train_time_ms": None if train_time_ms is None else int(train_time_ms),
    }


def load_metrics_from_stats_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return parse_metrics_from_stats(payload)


def resolve_manifest_stats_path(manifest_path: Path, cwd: Path) -> Path | None:
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return None
    candidates: list[str] = []
    stats_path = manifest.get("stats_path")
    if isinstance(stats_path, str) and stats_path.strip():
        candidates.append(stats_path)
    env = manifest.get("env")
    if isinstance(env, dict):
        for key in ("STATS_PATH", "METRICS_PATH"):
            value = env.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value)
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = (cwd / path).resolve()
        return path
    return None


def build_job_script(
    script_path: Path,
    exit_file: Path,
    cwd: Path,
    env: dict[str, str],
    command: str,
) -> None:
    lines = [
        "#!/bin/zsh",
        "set +e",
        f"cd {shlex.quote(str(cwd))}",
        *[f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items())],
        f"eval {shlex.quote(command)}",
        "ec=$?",
        f'print -r -- "$ec" > {shlex.quote(str(exit_file))}',
        'exit "$ec"',
    ]
    body = "\n".join(lines) + "\n"
    script_path.write_text(body, encoding="utf-8")
    script_path.chmod(0o755)


def running_job_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status = 'running'").fetchone()
    return int(row["n"])


def launch_pending_jobs(cfg: Config, conn: sqlite3.Connection) -> int:
    launched = 0
    pending_dir = cfg.runtime_dir / "queue" / "pending"
    running_dir = cfg.runtime_dir / "queue" / "running"
    pending = sorted(pending_dir.glob("*.json"))
    available_slots = max(cfg.max_concurrent_jobs - running_job_count(conn), 0)
    for manifest_path in pending[:available_slots]:
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:
            bad_target = cfg.runtime_dir / "queue" / "failed" / manifest_path.name
            shutil.move(str(manifest_path), bad_target)
            print(f"[loop] invalid manifest {manifest_path.name}: {exc}", file=sys.stderr)
            continue

        command = str(manifest.get("command", "")).strip()
        if not command:
            bad_target = cfg.runtime_dir / "queue" / "failed" / manifest_path.name
            shutil.move(str(manifest_path), bad_target)
            print(f"[loop] manifest missing command: {manifest_path.name}", file=sys.stderr)
            continue

        job_id = str(manifest.get("id") or manifest_path.stem)
        job_kind = str(manifest.get("job_kind") or "task")
        job_dir = cfg.runtime_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "run.log"
        exit_file = job_dir / "exit_code.txt"
        script_path = job_dir / "job.sh"
        env = {str(k): str(v) for k, v in dict(manifest.get("env", {})).items()}
        cwd = Path(str(manifest.get("cwd") or cfg.workdir))
        if not cwd.is_absolute():
            cwd = (cfg.workdir / cwd).resolve()
        timeout_seconds = int(manifest.get("timeout_seconds", cfg.default_job_timeout_seconds))
        description = str(manifest.get("description", job_id))

        running_manifest_path = running_dir / manifest_path.name
        shutil.move(str(manifest_path), running_manifest_path)
        dump_json(job_dir / "manifest.json", manifest)
        build_job_script(script_path, exit_file, cwd, env, command)

        with log_path.open("ab") as logf:
            proc = subprocess.Popen(
                [str(script_path)],
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        conn.execute(
            """
            INSERT OR REPLACE INTO jobs(
                job_id, job_kind, description, status, command, cwd, manifest_path, log_path, script_path,
                exit_file, pid, created_at, started_at, timeout_seconds
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_kind,
                description,
                command,
                str(cwd),
                str(running_manifest_path),
                str(log_path),
                str(script_path),
                str(exit_file),
                proc.pid,
                utc_now(),
                utc_now(),
                timeout_seconds,
            ),
        )
        conn.commit()
        append_journal_entry(
            cfg,
            f"Controller Job Launched: {job_id}",
            [
                f"Description: {description}",
                f"Job kind: {job_kind}",
                f"Command: {command}",
                f"CWD: {cwd}",
                f"Manifest: {running_manifest_path}",
                f"Log: {log_path}",
                f"Timeout seconds: {timeout_seconds}",
                f"PID: {proc.pid}",
            ],
        )
        launched += 1
        print(f"[loop] launched job {job_id} pid={proc.pid}")
    return launched


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def finalize_job(
    conn: sqlite3.Connection,
    cfg: Config,
    row: sqlite3.Row,
    final_status: str,
    exit_code: int | None,
) -> None:
    log_path = Path(str(row["log_path"]))
    manifest_path = Path(str(row["manifest_path"]))
    target_dir = cfg.runtime_dir / "queue" / ("done" if final_status == "done" else "failed")
    if manifest_path.exists():
        shutil.move(str(manifest_path), target_dir / manifest_path.name)
        manifest_path = target_dir / manifest_path.name

    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    stats_path = resolve_manifest_stats_path(manifest_path, Path(str(row["cwd"])))
    stats_metrics = load_metrics_from_stats_file(stats_path) if stats_path is not None else None
    log_metrics = parse_metrics_from_log(text)
    metrics = {
        key: (
            stats_metrics.get(key)
            if stats_metrics is not None and stats_metrics.get(key) is not None
            else log_metrics.get(key)
        )
        for key in ("val_bpb", "artifact_bytes", "train_time_ms")
    }
    conn.execute(
        """
        UPDATE jobs
        SET status = ?, finished_at = ?, exit_code = ?, val_bpb = ?, artifact_bytes = ?, train_time_ms = ?
        WHERE job_id = ?
        """,
        (
            final_status,
            utc_now(),
            exit_code,
            metrics["val_bpb"],
            metrics["artifact_bytes"],
            metrics["train_time_ms"],
            row["job_id"],
        ),
    )
    conn.commit()
    summary = {
        "job_id": row["job_id"],
        "job_kind": row["job_kind"] if "job_kind" in row.keys() else "task",
        "status": final_status,
        "exit_code": exit_code,
        "val_bpb": metrics["val_bpb"],
        "artifact_bytes": metrics["artifact_bytes"],
        "train_time_ms": metrics["train_time_ms"],
        "finished_at": utc_now(),
    }
    dump_json(cfg.runtime_dir / "jobs" / str(row["job_id"]) / "result.json", summary)
    append_journal_entry(
        cfg,
        f"Controller Job Result: {row['job_id']}",
        [
            f"Status: {final_status}",
            f"Job kind: {row['job_kind'] if 'job_kind' in row.keys() else 'task'}",
            f"Exit code: {exit_code}",
            f"Description: {row['description']}",
            f"Command: {row['command']}",
            f"val_bpb: {metrics['val_bpb'] if metrics['val_bpb'] is not None else 'n/a'}",
            f"artifact_bytes: {metrics['artifact_bytes'] if metrics['artifact_bytes'] is not None else 'n/a'}",
            f"train_time_ms: {metrics['train_time_ms'] if metrics['train_time_ms'] is not None else 'n/a'}",
            f"Manifest: {manifest_path}",
            f"Log: {log_path}",
        ],
    )
    print(f"[loop] finalized job {row['job_id']} status={final_status} val_bpb={metrics['val_bpb']}")


def poll_running_jobs(cfg: Config, conn: sqlite3.Connection) -> int:
    finished = 0
    rows = conn.execute("SELECT * FROM jobs WHERE status = 'running' ORDER BY started_at ASC").fetchall()
    for row in rows:
        exit_file = Path(str(row["exit_file"]))
        pid = int(row["pid"]) if row["pid"] is not None else None
        started_at = row["started_at"]
        timeout_seconds = int(row["timeout_seconds"])

        if exit_file.exists():
            try:
                exit_code = int(exit_file.read_text(encoding="utf-8").strip())
            except ValueError:
                exit_code = 1
            finalize_job(conn, cfg, row, "done" if exit_code == 0 else "failed", exit_code)
            finished += 1
            continue

        if pid is not None and started_at:
            started_ts = datetime.fromisoformat(str(started_at)).timestamp()
            if time.time() - started_ts > timeout_seconds:
                kill_process_group(pid)
                finalize_job(conn, cfg, row, "failed", -9)
                finished += 1
                continue

        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                finalize_job(conn, cfg, row, "failed", 1)
                finished += 1
    return finished


def best_local_score(conn: sqlite3.Connection, job_kind: str | None = None) -> sqlite3.Row | None:
    if job_kind is None:
        return conn.execute(
            """
            SELECT job_id, job_kind, val_bpb, artifact_bytes, train_time_ms, finished_at
            FROM jobs
            WHERE status = 'done' AND val_bpb IS NOT NULL
            ORDER BY val_bpb ASC, finished_at DESC
            LIMIT 1
            """
        ).fetchone()
    return conn.execute(
        """
        SELECT job_id, job_kind, val_bpb, artifact_bytes, train_time_ms, finished_at
        FROM jobs
        WHERE status = 'done' AND val_bpb IS NOT NULL AND COALESCE(job_kind, 'task') = ?
        ORDER BY val_bpb ASC, finished_at DESC
        LIMIT 1
        """,
        (job_kind,),
    ).fetchone()


def summarize_jobs(cfg: Config, conn: sqlite3.Connection) -> str:
    counts = {}
    for status in ("pending", "running", "done", "failed"):
        if status == "pending":
            counts[status] = len(list((cfg.runtime_dir / "queue" / "pending").glob("*.json")))
        else:
            row = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status = ?", (status,)).fetchone()
            counts[status] = int(row["n"])
    experiment_done = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE status = 'done' AND COALESCE(job_kind, 'task') = 'experiment'"
    ).fetchone()
    recent = conn.execute(
        """
        SELECT job_id, job_kind, status, val_bpb, artifact_bytes, train_time_ms, finished_at
        FROM jobs
        ORDER BY COALESCE(finished_at, started_at, created_at) DESC
        LIMIT 5
        """
    ).fetchall()
    lines = [
        f"Running jobs in DB: {counts['running']}",
        f"Completed jobs: {counts['done']}",
        f"Failed jobs: {counts['failed']}",
        f"Completed real experiments: {int(experiment_done['n'])}",
    ]
    pending_files = sorted((cfg.runtime_dir / "queue" / "pending").glob("*.json"))
    lines.append(f"Pending manifest files: {len(pending_files)}")
    lines.append(f"Staged manifest files: {staged_manifest_count(cfg)}")
    best = best_local_score(conn, "experiment")
    if best is None:
        lines.append("Best real experiment score: none yet")
    else:
        lines.append(
            f"Best real experiment score: {best['val_bpb']:.4f} "
            f"(job={best['job_id']} artifact={best['artifact_bytes']} train_time_ms={best['train_time_ms']})"
        )
    if recent:
        lines.append("Recent jobs:")
        for row in recent:
            score = "n/a" if row["val_bpb"] is None else f"{row['val_bpb']:.4f}"
            lines.append(
                f"- {row['job_id']} kind={row['job_kind'] if row['job_kind'] is not None else 'task'} "
                f"status={row['status']} val_bpb={score} "
                f"artifact={row['artifact_bytes']} train_time_ms={row['train_time_ms']}"
            )
    return "\n".join(lines)


def build_state_summary(cfg: Config, conn: sqlite3.Connection) -> str:
    queue_pending = len(list((cfg.runtime_dir / "queue" / "pending").glob("*.json")))
    queue_running = len(list((cfg.runtime_dir / "queue" / "running").glob("*.json")))
    queue_done = len(list((cfg.runtime_dir / "queue" / "done").glob("*.json")))
    queue_failed = len(list((cfg.runtime_dir / "queue" / "failed").glob("*.json")))
    turn = conn.execute(
        """
        SELECT turn_id, finished_at, summary, decision, best_score_bpb
        FROM codex_turns ORDER BY finished_at DESC LIMIT 1
        """
    ).fetchone()
    lines = [
        f"Workspace root: {cfg.workdir}",
        f"Journal file: {cfg.journal_file}",
        f"Manifest inbox dirs: {', '.join(str(p) for p in cfg.manifest_inbox_dirs)}",
        f"Pending manifests: {queue_pending}",
        f"Running manifests: {queue_running}",
        f"Done manifests: {queue_done}",
        f"Failed manifests: {queue_failed}",
        summarize_jobs(cfg, conn),
    ]
    active_turn = get_active_codex_turn(conn)
    if active_turn is not None:
        pid = active_turn.get("pid")
        alive = active_codex_turn_alive(active_turn)
        lines.append(
            f"Active Codex turn: {active_turn.get('turn_id')} "
            f"started_at={active_turn.get('started_at')} "
            f"pid={pid} status={'running' if alive else 'stale'}"
        )
        lines.append(f"Active Codex log: {active_turn.get('log_path')}")
        lines.append(f"Active Codex output target: {active_turn.get('output_path')}")
    if turn is None:
        lines.append("Last Codex turn: none yet")
    else:
        lines.append(
            f"Last Codex turn: {turn['turn_id']} decision={turn['decision']} "
            f"best_score_bpb={turn['best_score_bpb']}"
        )
        lines.append(f"Last Codex summary: {turn['summary']}")
    return "\n".join(lines)


def build_prompt(cfg: Config, leaderboard_summary: str, state_summary: str) -> str:
    template = cfg.prompt_template_file.read_text(encoding="utf-8")
    goal = cfg.goal_file.read_text(encoding="utf-8")
    repo_context_section = ""
    if cfg.repo_context_file is not None and cfg.repo_context_file.exists():
        repo_context = cfg.repo_context_file.read_text(encoding="utf-8").strip()
        if repo_context:
            repo_context_section = f"\n## Historical Repo Context\n\n{repo_context}\n"
    return template.format(
        goal=goal.strip(),
        leaderboard_summary=leaderboard_summary.strip(),
        state_summary=state_summary.strip(),
        repo_context_section=repo_context_section.rstrip(),
    )


def invoke_codex(cfg: Config, conn: sqlite3.Connection, leaderboard_summary: str, state_summary: str) -> dict[str, Any]:
    turn_id = datetime.now(timezone.utc).strftime("turn_%Y%m%dT%H%M%SZ")
    prompt_path = cfg.runtime_dir / "reports" / f"{turn_id}.prompt.md"
    output_path = cfg.runtime_dir / "reports" / f"{turn_id}.output.json"
    log_path = cfg.runtime_dir / "logs" / f"{turn_id}.codex.log"
    prompt = build_prompt(cfg, leaderboard_summary, state_summary)
    prompt_path.write_text(prompt, encoding="utf-8")

    cmd = [
        "codex",
        "exec",
        *cfg.codex_args,
        "-m",
        cfg.codex_model,
        "--output-schema",
        str(cfg.agent_output_schema_file),
        "-o",
        str(output_path),
        "-C",
        str(cfg.workdir),
        "-",
    ]
    started_at = utc_now()
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=cfg.workdir,
                text=True,
            )
            set_active_codex_turn(
                conn,
                turn_id=turn_id,
                started_at=started_at,
                prompt_path=prompt_path,
                output_path=output_path,
                log_path=log_path,
                pid=proc.pid,
            )
            write_heartbeat(cfg, conn, f"codex-running:{turn_id}")
            try:
                proc.communicate(prompt, timeout=cfg.codex_timeout_seconds)
                return_code = int(proc.returncode or 0)
                timed_out = False
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                timed_out = True
                return_code = 124
            finally:
                clear_active_codex_turn(conn)
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = 124
        clear_active_codex_turn(conn)
    finished_at = utc_now()

    if output_path.exists():
        try:
            output = load_json(output_path)
        except Exception:
            output = {
                "summary": "Codex wrote an unreadable output file.",
                "decision": "continue",
                "sleep_seconds": cfg.default_sleep_seconds,
                "best_score_bpb": None,
                "best_score_source": None,
                "job_manifest_created": False,
                "job_manifest_path": None,
                "repo_changed": False,
                "needs_human_attention": True,
                "human_message": "Invalid structured output from codex exec.",
            }
    else:
        output = {
            "summary": "Codex did not produce a structured output file." if not timed_out else "Codex turn timed out before producing structured output.",
            "decision": "continue",
            "sleep_seconds": cfg.default_sleep_seconds,
            "best_score_bpb": None,
            "best_score_source": None,
            "job_manifest_created": False,
            "job_manifest_path": None,
            "repo_changed": False,
            "needs_human_attention": True,
            "human_message": "Missing structured output from codex exec.",
        }

    conn.execute(
        """
        INSERT INTO codex_turns(
            turn_id, started_at, finished_at, exit_code, prompt_path, output_path, log_path,
            summary, decision, sleep_seconds, best_score_bpb, needs_human_attention
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            turn_id,
            started_at,
            finished_at,
            return_code,
            str(prompt_path),
            str(output_path),
            str(log_path),
            str(output.get("summary", "")),
            str(output.get("decision", "continue")),
            int(output.get("sleep_seconds", cfg.default_sleep_seconds)),
            output.get("best_score_bpb"),
            1 if bool(output.get("needs_human_attention")) else 0,
        ),
    )
    conn.commit()
    append_journal_entry(
        cfg,
        f"Controller Turn Summary: {turn_id}",
        [
            f"Codex exit code: {return_code}",
            f"Decision: {output.get('decision')}",
            f"Summary: {output.get('summary', '')}",
            f"Best score bpb: {output.get('best_score_bpb')}",
            f"Best score source: {output.get('best_score_source')}",
            f"Job manifest created: {output.get('job_manifest_created')}",
            f"Job manifest path: {output.get('job_manifest_path')}",
            f"Repo changed: {output.get('repo_changed')}",
            f"Needs human attention: {output.get('needs_human_attention')}",
            f"Human message: {output.get('human_message', '')}",
            f"Prompt: {prompt_path}",
            f"Log: {log_path}",
        ],
    )
    print(f"[loop] codex turn {turn_id} exit={return_code} decision={output.get('decision')}")
    return output


def write_heartbeat(cfg: Config, conn: sqlite3.Connection, note: str) -> None:
    best = best_local_score(conn, "experiment")
    active_turn = get_active_codex_turn(conn)
    payload = {
        "timestamp": utc_now(),
        "note": note,
        "best_local_score_bpb": None if best is None else best["val_bpb"],
        "running_jobs": running_job_count(conn),
        "pending_manifests": len(list((cfg.runtime_dir / "queue" / "pending").glob("*.json"))),
        "staged_manifests": staged_manifest_count(cfg),
        "active_codex_turn": active_turn,
    }
    dump_json(cfg.runtime_dir / "heartbeat.json", payload)


def controller_cycle(cfg: Config, conn: sqlite3.Connection) -> int:
    if clear_stale_active_codex_turn(conn):
        write_heartbeat(cfg, conn, "cleared-stale-codex-turn")
    leaderboard = refresh_leaderboard_if_needed(cfg, conn)
    leaderboard_summary = summarize_leaderboard(leaderboard, cfg)
    ingest_manifest_inboxes(cfg, conn)
    poll_running_jobs(cfg, conn)
    launch_pending_jobs(cfg, conn)
    write_heartbeat(cfg, conn, "post-poll")

    if running_job_count(conn) > 0:
        maybe_checkpoint_repo(cfg, "job-launch-or-poll")
        return min(cfg.default_sleep_seconds, 30)

    if len(list((cfg.runtime_dir / "queue" / "pending").glob("*.json"))) > 0:
        launch_pending_jobs(cfg, conn)
        if running_job_count(conn) > 0:
            maybe_checkpoint_repo(cfg, "pending-job-launch")
            return min(cfg.default_sleep_seconds, 30)

    best = best_local_score(conn, "experiment")
    if cfg.stop_when_best_bpb_le is not None and best is not None and float(best["val_bpb"]) <= cfg.stop_when_best_bpb_le:
        print(f"[loop] stop threshold reached: {best['val_bpb']:.4f} <= {cfg.stop_when_best_bpb_le:.4f}")
        maybe_checkpoint_repo(cfg, "stop-threshold")
        return -1

    state_summary = build_state_summary(cfg, conn)
    output = invoke_codex(cfg, conn, leaderboard_summary, state_summary)
    write_heartbeat(cfg, conn, "post-codex")
    maybe_checkpoint_repo(cfg, f"codex-{output.get('decision', 'continue')}")

    decision = str(output.get("decision", "continue"))
    if decision.startswith("halt") and not cfg.allow_agent_halt:
        return max(cfg.default_sleep_seconds, int(output.get("sleep_seconds", cfg.default_sleep_seconds)))
    return max(5, int(output.get("sleep_seconds", cfg.default_sleep_seconds)))


def print_status(cfg: Config, conn: sqlite3.Connection) -> None:
    leaderboard = conn.execute(
        "SELECT * FROM leaderboard_snapshots ORDER BY fetched_at DESC LIMIT 1"
    ).fetchone()
    print("Codex Loop Status")
    print("-----------------")
    pid_path = cfg.runtime_dir / "controller.pid"
    if pid_path.exists():
        pid = pid_path.read_text(encoding="utf-8").strip()
        alive = False
        if pid:
            try:
                os.kill(int(pid), 0)
                alive = True
            except Exception:
                alive = False
        print(f"Controller PID: {pid} ({'running' if alive else 'stale'})")
    else:
        print("Controller PID: not recorded")
    print(f"Runtime dir: {cfg.runtime_dir}")
    print()
    print(summarize_leaderboard(dict(leaderboard) if leaderboard is not None else None, cfg))
    print()
    print(build_state_summary(cfg, conn))


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent Codex supervisor loop")
    parser.add_argument("command", choices=["start", "once", "status"], help="controller mode")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to config JSON")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    ensure_runtime_layout(cfg.runtime_dir)
    ensure_manifest_inboxes(cfg)
    conn = open_db(cfg.runtime_dir)

    if args.command == "status":
        print_status(cfg, conn)
        return

    if args.command == "once":
        sleep_seconds = controller_cycle(cfg, conn)
        print(f"[loop] cycle complete; suggested sleep={sleep_seconds}")
        return

    if args.command == "start":
        pid_path = cfg.runtime_dir / "controller.pid"
        pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        print(f"[loop] starting controller in {cfg.workdir}")
        while True:
            try:
                sleep_seconds = controller_cycle(cfg, conn)
                if sleep_seconds < 0:
                    print("[loop] controller exiting due to stop condition")
                    return
            except KeyboardInterrupt:
                print("[loop] interrupted")
                return
            except Exception as exc:
                print(f"[loop] controller error: {exc}", file=sys.stderr)
                sleep_seconds = max(cfg.default_sleep_seconds, 60)
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
