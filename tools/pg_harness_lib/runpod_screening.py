from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pg_harness_lib.constants import ROOT
from pg_harness_lib.presets import build_command, shell_env_command, tracked_env
from runpod_client import RunpodClient, RunpodError
from runpod_config import (
    DEFAULT_CONTAINER_DISK_GB,
    DEFAULT_PORTS,
    DEFAULT_SCREENING_GPU_TYPE_ID,
    DEFAULT_SCREENING_POD_NAME,
    DEFAULT_SCREENING_TEMPLATE_ID,
    DEFAULT_VOLUME_GB,
    DEFAULT_VOLUME_MOUNT_PATH,
)


REMOTE_ROOT = Path("/workspace/parameter-golf")
REMOTE_LOGS_DIR = REMOTE_ROOT / "logs"
RESUME_RETRY_SLEEP_SECONDS = 15
RESUME_RETRY_ATTEMPTS = 12
RETRYABLE_CAPACITY_SNIPPETS = (
    "not enough free gpus",
    "not enough free gpus on the host machine",
)
RSYNC_EXCLUDES = (
    ".git",
    ".env",
    ".env.*",
    ".venv",
    "__pycache__",
    ".DS_Store",
    ".mypy_cache",
    ".code",
    "logs",
    "experiments/runs",
    "data/datasets",
    "data/tokenizers",
)


def _find_pod_by_name(pods: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matches = [pod for pod in pods if pod.get("name") == name]
    if not matches:
        return None
    if len(matches) > 1:
        ids = ", ".join(pod["id"] for pod in matches)
        raise RunpodError(f"Multiple pods found with name {name!r}: {ids}")
    return matches[0]


class RunpodScreeningSession:
    def __init__(
        self,
        *,
        keep_pod_running: bool = False,
        pod_name: str = DEFAULT_SCREENING_POD_NAME,
        gpu_type_id: str = DEFAULT_SCREENING_GPU_TYPE_ID,
        template_id: str = DEFAULT_SCREENING_TEMPLATE_ID,
        container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
        volume_gb: int = DEFAULT_VOLUME_GB,
        identity_file: str = "~/.ssh/id_rsa",
    ):
        self.keep_pod_running = keep_pod_running
        self.pod_name = pod_name
        self.gpu_type_id = gpu_type_id
        self.template_id = template_id
        self.container_disk_gb = container_disk_gb
        self.volume_gb = volume_gb
        self.identity_file = str(Path(identity_file).expanduser())
        self.client = RunpodClient.from_env()
        self.pod: dict[str, Any] | None = None
        self._synced = False
        self._bootstrapped_variants: set[tuple[str, int | None]] = set()

    def __enter__(self) -> "RunpodScreeningSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.keep_pod_running and self.pod is not None:
            self.client.stop_pod(self.pod["id"])

    def _is_retryable_capacity_error(self, exc: RunpodError) -> bool:
        message = str(exc).lower()
        return any(snippet in message for snippet in RETRYABLE_CAPACITY_SNIPPETS)

    def _create_pod_once(self) -> dict[str, Any]:
        payload = {
            "cloudType": "SECURE",
            "computeType": "GPU",
            "containerDiskInGb": self.container_disk_gb,
            "gpuCount": 1,
            "gpuTypeIds": [self.gpu_type_id],
            "interruptible": False,
            "name": self.pod_name,
            "ports": list(DEFAULT_PORTS),
            "supportPublicIp": True,
            "templateId": self.template_id,
            "volumeInGb": self.volume_gb,
            "volumeMountPath": DEFAULT_VOLUME_MOUNT_PATH,
        }
        return self.client.create_pod(payload)

    def _create_pod(self) -> dict[str, Any]:
        last_error: RunpodError | None = None
        for attempt in range(RESUME_RETRY_ATTEMPTS):
            try:
                return self._create_pod_once()
            except RunpodError as exc:
                if not self._is_retryable_capacity_error(exc):
                    raise
                last_error = exc
                if attempt + 1 == RESUME_RETRY_ATTEMPTS:
                    break
                time.sleep(RESUME_RETRY_SLEEP_SECONDS)
        raise RunpodError(
            "Runpod could not allocate a screening pod after repeated capacity retries. "
            f"Last error: {last_error}"
        )

    def _resume_pod(self, pod_id: str) -> dict[str, Any]:
        last_error: RunpodError | None = None
        for attempt in range(RESUME_RETRY_ATTEMPTS):
            try:
                return self.client.resume_pod(pod_id)
            except RunpodError as exc:
                if not self._is_retryable_capacity_error(exc):
                    raise
                last_error = exc
                if attempt + 1 == RESUME_RETRY_ATTEMPTS:
                    break
                time.sleep(RESUME_RETRY_SLEEP_SECONDS)
        raise RunpodError(
            f"Runpod could not resume screening pod {pod_id} after repeated capacity retries. "
            f"Last error: {last_error}"
        )

    def ensure_pod(self) -> dict[str, Any]:
        if self.pod is not None and self.pod.get("desiredStatus") == "RUNNING" and self._has_ssh_endpoint(self.pod):
            return self.pod

        pod = _find_pod_by_name(list(self.client.list_pods()), self.pod_name)
        if pod is None:
            pod = self._create_pod()
        elif pod.get("desiredStatus") != "RUNNING":
            pod = self._resume_pod(pod["id"])

        deadline = time.time() + 300
        while time.time() < deadline:
            refreshed = self.client.get_pod(pod["id"])
            if refreshed.get("desiredStatus") == "RUNNING" and self._has_ssh_endpoint(refreshed):
                self.pod = refreshed
                return self.pod
            time.sleep(5)
        raise RunpodError(f"Timed out waiting for pod {pod['id']} to become SSH-ready.")

    def _has_ssh_endpoint(self, pod: dict[str, Any]) -> bool:
        public_ip = pod.get("publicIp")
        port_mappings = pod.get("portMappings") or {}
        ssh_port = port_mappings.get("22") or port_mappings.get(22)
        return bool(public_ip and ssh_port)

    def _ssh_base(self) -> list[str]:
        pod = self.ensure_pod()
        port_mappings = pod.get("portMappings") or {}
        ssh_port = port_mappings.get("22") or port_mappings.get(22)
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "LogLevel=ERROR",
            "-i",
            self.identity_file,
            "-p",
            str(ssh_port),
            self._remote_host(),
        ]

    def _ssh_shell(self, command: str) -> list[str]:
        return [*self._ssh_base(), f"bash -lc {shlex.quote(command)}"]

    def _remote_host(self) -> str:
        pod = self.ensure_pod()
        return f"root@{pod['publicIp']}"

    def _ssh_transport(self) -> list[str]:
        pod = self.ensure_pod()
        port_mappings = pod.get("portMappings") or {}
        ssh_port = port_mappings.get("22") or port_mappings.get(22)
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "LogLevel=ERROR",
            "-i",
            self.identity_file,
            "-p",
            str(ssh_port),
        ]

    def sync_workspace(self) -> None:
        if self._synced:
            return
        if shutil.which("rsync") is None:
            raise RunpodError("rsync is required locally to sync the workspace to Runpod.")
        last_error: Exception | None = None
        for _ in range(30):
            try:
                subprocess.run(self._ssh_shell(f"mkdir -p {shlex.quote(str(REMOTE_ROOT))}"), check=True)
                rsync_cmd = [
                    "rsync",
                    "-az",
                    "--delete",
                    "--no-owner",
                    "--no-group",
                    "-e",
                    " ".join(shlex.quote(bit) for bit in self._ssh_transport()),
                ]
                for pattern in RSYNC_EXCLUDES:
                    rsync_cmd.extend(["--exclude", pattern])
                rsync_cmd.extend([f"{ROOT}/", f"{self._remote_host()}:{REMOTE_ROOT}/"])
                subprocess.run(rsync_cmd, check=True)
                self._synced = True
                return
            except subprocess.CalledProcessError as exc:
                last_error = exc
                time.sleep(5)
        raise RunpodError(f"Timed out syncing the workspace to the screening pod: {last_error}")

    def _run_ssh_with_retry(self, command: list[str], *, attempts: int = 20) -> None:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                subprocess.run(command, check=True)
                return
            except subprocess.CalledProcessError as exc:
                last_error = exc
                time.sleep(5)
        raise RunpodError(f"Remote command failed after retries: {last_error}")

    def ensure_dataset(self, preset: dict[str, Any]) -> None:
        remote = preset.get("remote", {})
        data = remote.get("data")
        if not data:
            return
        variant = str(data["variant"])
        train_shards = data.get("train_shards")
        bootstrap_key = (variant, int(train_shards) if train_shards is not None else None)
        if bootstrap_key in self._bootstrapped_variants:
            return

        download = f"python3 data/cached_challenge_fineweb.py --variant {shlex.quote(variant)}"
        if train_shards is not None:
            download += f" --train-shards {int(train_shards)}"
        command = f"cd {shlex.quote(str(REMOTE_ROOT))} && {download}"
        self._run_ssh_with_retry(self._ssh_shell(command))
        self._bootstrapped_variants.add(bootstrap_key)

    def run(
        self,
        *,
        run_dir: Path,
        preset_name: str,
        preset: dict[str, Any],
        overrides: dict[str, str],
    ) -> int:
        del preset_name
        self.ensure_pod()
        self.sync_workspace()
        self.ensure_dataset(preset)

        tracked = tracked_env(preset, overrides)
        remote_command = build_command(REMOTE_ROOT, preset)
        remote_log_path = REMOTE_LOGS_DIR / f"{run_dir.name}.log"
        remote_shell = "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(str(REMOTE_LOGS_DIR))}",
                f"cd {shlex.quote(str(REMOTE_ROOT))}",
                f"{shell_env_command(tracked, remote_command)} 2>&1 | tee {shlex.quote(str(remote_log_path))}",
            ]
        )
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                self._ssh_shell(remote_shell),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        return proc.returncode
