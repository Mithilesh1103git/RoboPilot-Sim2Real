"""Process-backed job manager for the standalone training application.

Jobs are launched as independent Python subprocesses. This module intentionally
imports no runtime application code and does not call the runtime API.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "training"


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.RLock()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_locked()
            return [
                self._public(job)
                for job in sorted(
                    self._jobs.values(), key=lambda x: x["created_at"], reverse=True
                )
            ]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._refresh_locked()
            job = self._jobs.get(job_id)
            return self._public(job) if job else None

    def start(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        command = self._command(kind, params)
        job_id = str(uuid.uuid4())
        created = time.time()
        record: dict[str, Any] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": created,
            "started_at": None,
            "finished_at": None,
            "command": command,
            "params": params,
            "exit_code": None,
            "logs": [],
            "error": None,
            "artifacts": [],
        }
        with self._lock:
            self._jobs[job_id] = record
        thread = threading.Thread(
            target=self._launch, args=(job_id, command), daemon=True
        )
        thread.start()
        return self.get(job_id) or self._public(record)

    def stop(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            proc = self._processes.get(job_id)
            if not job:
                return None
            if proc and proc.poll() is None:
                job["status"] = "stopping"
                proc.terminate()
                threading.Thread(
                    target=self._kill_if_needed, args=(job_id, proc), daemon=True
                ).start()
            elif job["status"] == "queued":
                job["status"] = "cancelled"
                job["finished_at"] = time.time()
            return self._public(job)

    def _command(self, kind: str, params: dict[str, Any]) -> list[str]:
        py = sys.executable
        if kind == "rl_train":
            return [
                py,
                "-m",
                "training.rl.train",
                "--algorithm",
                params["algorithm"],
                "--timesteps",
                str(params["timesteps"]),
                "--seed",
                str(params["seed"]),
                "--device",
                params["device"],
                "--robot-name",
                params.get("robot_name", "panda"),
                "--env-id",
                params.get("env_id", "RoboPilotPickPlace-v0"),
                "--output",
                params["output"],
            ]
        if kind == "synthetic_collect":
            return [
                py,
                "-m",
                "training.synthetic.collect_episodes",
                "--episodes",
                str(params["episodes"]),
                "--seed",
                str(params["seed"]),
                "--output",
                params["output"],
                "--robot-name",
                params.get("robot_name", "panda"),
            ]
        if kind == "evaluate":
            return [
                py,
                "-m",
                "training.rl.evaluate",
                "--checkpoint",
                params["checkpoint"],
                "--algorithm",
                params["algorithm"],
                "--episodes",
                str(params["episodes"]),
                "--seed",
                str(params["seed"]),
                "--robot-name",
                params.get("robot_name", "panda"),
                "--env-id",
                params.get("env_id", "RoboPilotPickPlace-v0"),
            ]
        if kind == "world_model_train":
            return [
                py,
                "-m",
                "training.rl.train_world_model",
                "--dataset",
                params["dataset"],
                "--output",
                params["output"],
                "--epochs",
                str(params["epochs"]),
            ]
        raise ValueError(f"Unsupported job type: {kind}")

    def _launch(self, job_id: str, command: list[str]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] == "cancelled":
                return
            job["status"] = "starting"
            job["started_at"] = time.time()
        env = os.environ.copy()
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (
            os.pathsep + old_pythonpath if old_pythonpath else ""
        )
        try:
            proc = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._processes[job_id] = proc
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = "running"
            assert proc.stdout is not None
            for line in proc.stdout:
                self._append_log(job_id, line.rstrip("\r\n"))
            code = proc.wait()
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job["exit_code"] = code
                    job["finished_at"] = time.time()
                    if job["status"] == "stopping":
                        job["status"] = "cancelled"
                    else:
                        job["status"] = "succeeded" if code == 0 else "failed"
                    if code != 0 and job["status"] == "failed":
                        job["error"] = (
                            f"Training process exited with code {code}. Inspect logs for details."
                        )
                    job["artifacts"] = self._discover_artifacts(job)
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["finished_at"] = time.time()
                    job["error"] = str(exc)
                    job["logs"].append(f"Job launcher error: {exc}")
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["logs"].append(line)
                # Bound memory use while keeping the latest useful output.
                if len(job["logs"]) > 2000:
                    del job["logs"][: len(job["logs"]) - 2000]

    def _discover_artifacts(self, job: dict[str, Any]) -> list[str]:
        values = []
        for key in ("output",):
            value = job.get("params", {}).get(key)
            if value:
                path = Path(value)
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                candidates = [path]
                if path.suffix != ".zip":
                    candidates.append(Path(str(path) + ".zip"))
                for candidate in candidates:
                    if candidate.exists():
                        try:
                            values.append(
                                candidate.resolve()
                                .relative_to(PROJECT_ROOT.resolve())
                                .as_posix()
                            )
                        except ValueError:
                            values.append(str(candidate.resolve()))
        return list(dict.fromkeys(values))

    def _refresh_locked(self) -> None:
        # Process state is finalized by the launcher thread; this is intentionally
        # a no-op hook for future persistence or worker reconciliation.
        return

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in job.items() if k != "logs"} | {
            "log_count": len(job.get("logs", []))
        }

    def logs(self, job_id: str, offset: int = 0) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            logs = job["logs"]
            start = max(0, min(int(offset), len(logs)))
            return {
                "job_id": job_id,
                "offset": start,
                "next_offset": len(logs),
                "lines": logs[start:],
            }

    def _kill_if_needed(self, job_id: str, proc: subprocess.Popen[str]) -> None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


manager = JobManager()
