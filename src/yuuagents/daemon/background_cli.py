#!/usr/bin/env python3
"""background — run long-lived commands in the background inside a container.

Usage:
    background run <command>           Start command, print {"id","status"}
    background tail <id> [--lines N]   Last N lines of output (default 20)
    background drain <id>              Full output if done, else current buffer + status
    background wait <id> [<id> ...]    Block until task(s) finish, print results
    background kill <id>               Kill a running task
    background list                    List all tasks
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid

BG_DIR = os.environ.get("BACKGROUND_DIR", "/tmp/.background")


def _task_dir(task_id: str) -> str:
    return os.path.join(BG_DIR, task_id)


def _ensure_dir():
    os.makedirs(BG_DIR, exist_ok=True)


def _read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _task_status(task_id: str) -> dict:
    d = _task_dir(task_id)
    if not os.path.isdir(d):
        return {"id": task_id, "error": "not found"}
    pid_str = _read_file(os.path.join(d, "pid")).strip()
    cmd = _read_file(os.path.join(d, "cmd")).strip()
    rc_file = os.path.join(d, "rc")
    if os.path.exists(rc_file):
        rc = int(_read_file(rc_file).strip())
        return {"id": task_id, "status": "done", "exit_code": rc, "command": cmd}
    # check if process still alive
    if pid_str:
        try:
            os.kill(int(pid_str), 0)
        except (ProcessLookupError, ValueError):
            # process gone but no rc file — crashed
            return {"id": task_id, "status": "done", "exit_code": -1, "command": cmd}
    return {"id": task_id, "status": "running", "command": cmd}


def cmd_run(args: list[str]):
    if not args:
        print("Usage: background run <command>", file=sys.stderr)
        sys.exit(1)
    command = " ".join(args)
    _ensure_dir()
    task_id = f"bg-{uuid.uuid4().hex[:8]}"
    d = _task_dir(task_id)
    os.makedirs(d)
    with open(os.path.join(d, "cmd"), "w") as f:
        f.write(command)
    out_path = os.path.join(d, "out")
    rc_path = os.path.join(d, "rc")
    # fork a daemon process
    pid = os.fork()
    if pid == 0:
        # child — detach
        os.setsid()
        # redirect stdout/stderr to file
        with open(out_path, "w") as out_f:
            try:
                proc = subprocess.run(
                    command, shell=True,
                    stdout=out_f, stderr=subprocess.STDOUT,
                    cwd=os.environ.get("HOME", "/"),
                )
                with open(rc_path, "w") as rc_f:
                    rc_f.write(str(proc.returncode))
            except Exception as e:
                out_f.write(f"\n[background error: {e}]\n")
                with open(rc_path, "w") as rc_f:
                    rc_f.write("-1")
        os._exit(0)
    else:
        # parent — write pid and return
        with open(os.path.join(d, "pid"), "w") as f:
            f.write(str(pid))
        print(json.dumps({"id": task_id, "status": "running"}))


def cmd_tail(args: list[str]):
    if not args:
        print("Usage: background tail <id> [--lines N]", file=sys.stderr)
        sys.exit(1)
    task_id = args[0]
    n = 20
    if len(args) >= 3 and args[1] == "--lines":
        n = int(args[2])
    d = _task_dir(task_id)
    out_path = os.path.join(d, "out")
    content = _read_file(out_path)
    lines = content.splitlines()
    tail = "\n".join(lines[-n:])
    status = _task_status(task_id)
    print(json.dumps({"id": task_id, "status": status["status"], "tail": tail}))


def cmd_drain(args: list[str]):
    if not args:
        print("Usage: background drain <id>", file=sys.stderr)
        sys.exit(1)
    task_id = args[0]
    d = _task_dir(task_id)
    out_path = os.path.join(d, "out")
    content = _read_file(out_path)
    status = _task_status(task_id)
    result = {"id": task_id, "status": status["status"], "output": content}
    if "exit_code" in status:
        result["exit_code"] = status["exit_code"]
    print(json.dumps(result))


def cmd_wait(args: list[str]):
    if not args:
        print("Usage: background wait <id> [<id> ...]", file=sys.stderr)
        sys.exit(1)
    results = []
    for task_id in args:
        while True:
            st = _task_status(task_id)
            if st.get("status") == "done" or "error" in st:
                break
            time.sleep(1)
        d = _task_dir(task_id)
        out_path = os.path.join(d, "out")
        content = _read_file(out_path)
        r = {"id": task_id, "status": st.get("status", "error"), "output": content}
        if "exit_code" in st:
            r["exit_code"] = st["exit_code"]
        results.append(r)
    print(json.dumps(results if len(results) > 1 else results[0]))


def cmd_kill(args: list[str]):
    if not args:
        print("Usage: background kill <id>", file=sys.stderr)
        sys.exit(1)
    task_id = args[0]
    d = _task_dir(task_id)
    pid_str = _read_file(os.path.join(d, "pid")).strip()
    if not pid_str:
        print(json.dumps({"id": task_id, "error": "no pid"}))
        return
    try:
        os.killpg(int(pid_str), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    # write rc if not already done
    rc_path = os.path.join(d, "rc")
    if not os.path.exists(rc_path):
        with open(rc_path, "w") as f:
            f.write("-15")
    print(json.dumps({"id": task_id, "status": "killed"}))


def cmd_list(_args: list[str]):
    _ensure_dir()
    tasks = []
    for name in sorted(os.listdir(BG_DIR)):
        if name.startswith("bg-"):
            tasks.append(_task_status(name))
    print(json.dumps(tasks))


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    dispatch = {
        "run": cmd_run,
        "tail": cmd_tail,
        "drain": cmd_drain,
        "wait": cmd_wait,
        "kill": cmd_kill,
        "list": cmd_list,
    }
    fn = dispatch.get(cmd)
    if fn is None:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    fn(rest)


if __name__ == "__main__":
    main()
