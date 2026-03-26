#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a command in a detached double-forked daemon.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        raise SystemExit("detach_run.py requires a command after --")

    stdout_path = args.stdout.expanduser().resolve()
    stderr_path = (args.stderr or args.stdout).expanduser().resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    pid = os.fork()
    if pid > 0:
        print(pid)
        return

    os.setsid()

    pid = os.fork()
    if pid > 0:
        os._exit(0)

    os.chdir(args.cwd.expanduser().resolve())
    os.umask(0o022)

    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    sys.stdout.flush()
    sys.stderr.flush()

    with open("/dev/null", "rb", buffering=0) as devnull_r:
        with open(stdout_path, "ab", buffering=0) as stdout_f:
            with open(stderr_path, "ab", buffering=0) as stderr_f:
                os.dup2(devnull_r.fileno(), 0)
                os.dup2(stdout_f.fileno(), 1)
                os.dup2(stderr_f.fileno(), 2)
                subprocess.Popen(
                    args.cmd,
                    close_fds=True,
                    start_new_session=True,
                    cwd=args.cwd.expanduser().resolve(),
                )
    os._exit(0)


if __name__ == "__main__":
    main()
