#!/usr/bin/env python3
"""Operational sbatch wrapper; never changes scientific configurations.

Install as ``scheduler/bin/sbatch`` in one round's private PATH. Controllers
and their descendants inherit that PATH. CPU submissions retain kempner_dev;
authorized Kempner GPU partitions choose the larger current account fairshare.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ACCOUNTS = ("kempner_dev", "kempner_bsabatini_lab")
GPU_PARTITIONS = {"kempner_requeue", "kempner_h100"}


def choose_account(output):
    values = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) >= 4 and fields[0] in ACCOUNTS and fields[1] and not fields[2]:
            values[fields[0]] = float(fields[3])
    if set(values) != set(ACCOUNTS):
        raise ValueError("Missing default account fairshare association")
    return max(ACCOUNTS, key=lambda account: values[account]), values


def protect_confirmation(args, root, action, study):
    """Keep timed confirmation cohorts off preemptible partitions."""
    config = root / study / "config.json"
    if action != "train" or not config.exists():
        return args, False
    if json.loads(config.read_text()).get("stage") != "joint_confirmation":
        return args, False
    hardware = next((a.split("=", 1)[1] for a in args if a.startswith("--constraint=")), "")
    partition, qos = {"h100": ("kempner_h100_priority", "kemp_gpu16_id38"),
                      "h200": ("kempner_eng", "normal")}[hardware]
    args = [a for a in args if not a.startswith(("--partition=", "--qos="))]
    return ["--partition=" + partition, "--qos=" + qos, *args], True


def main():
    args = sys.argv[1:]
    root = Path(os.environ["NDFA_ACCOUNT_ROUTE_ROOT"]).resolve()
    # Scope this wrapper to the exact experiment, not other user submissions.
    if str(root / "run.sbatch") not in args:
        os.execv("/usr/bin/sbatch", ["sbatch", *args])
    args, protected = protect_confirmation(
        args, root, os.environ.get("NDFA_INTEGRATED_ACTION"),
        os.environ.get("NDFA_INTEGRATED_STUDY", "all"),
    )
    partition = next((a.split("=", 1)[1] for a in args if a.startswith("--partition=")), "")
    account, values, error = "kempner_dev", {}, None
    if partition in GPU_PARTITIONS:
        try:
            result = subprocess.run(
                ["sshare", "-nP", "-A", ",".join(ACCOUNTS), "-u", os.environ["USER"],
                 "-o", "Account,User,Partition,FairShare"],
                capture_output=True, text=True, check=True, timeout=20,
            )
            account, values = choose_account(result.stdout)
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            # Queue availability must not stop the frozen scientific workflow.
            # Fall back to the last verified account and disclose the query error.
            account = "kempner_bsabatini_lab"
            error = str(exc)
    args = [a for a in args if not a.startswith("--account=")]
    result = subprocess.run(["/usr/bin/sbatch", "--account=" + account, *args], capture_output=True, text=True)
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), account=account,
                   protected_confirmation=protected,
                   partition=partition, fairshare=values, fairshare_error=error,
                   command=["/usr/bin/sbatch", "--account=" + account, *args],
                   returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    folder = root / "scheduler" / "submissions"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (uuid.uuid4().hex + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
