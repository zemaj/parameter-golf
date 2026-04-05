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
from runpod_lifecycle import (
    RESUME_RETRY_ATTEMPTS,
    RESUME_RETRY_SLEEP_SECONDS,
    build_screening_pod_payload,
    ensure_screening_pod_by_id,
    ensure_named_screening_pod,
)
from runpod_config import (
    DEFAULT_CONTAINER_DISK_GB,
    DEFAULT_SCREENING_GPU_TYPE_ID,
    DEFAULT_SCREENING_POD_ID,
    DEFAULT_SCREENING_POD_NAME,
    DEFAULT_SCREENING_TEMPLATE_ID,
    DEFAULT_VOLUME_GB,
)


REMOTE_ROOT = Path("/workspace/parameter-golf")
REMOTE_LOGS_DIR = REMOTE_ROOT / "logs"
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


class RunpodScreeningSession:
    def __init__(
        self,
        *,
        keep_pod_running: bool = False,
        pod_id: str = DEFAULT_SCREENING_POD_ID,
        pod_name: str = DEFAULT_SCREENING_POD_NAME,
        gpu_type_id: str = DEFAULT_SCREENING_GPU_TYPE_ID,
        template_id: str = DEFAULT_SCREENING_TEMPLATE_ID,
        container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
        volume_gb: int = DEFAULT_VOLUME_GB,
        identity_file: str = "~/.ssh/id_rsa",
    ):
        self.keep_pod_running = keep_pod_running
        self.pod_id = pod_id.strip()
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

    def ensure_pod(self) -> dict[str, Any]:
        if self.pod is not None and self.pod.get("desiredStatus") == "RUNNING" and self._has_ssh_endpoint(self.pod):
            return self.pod

        if self.pod_id:
            pod = ensure_screening_pod_by_id(
                self.client,
                pod_id=self.pod_id,
                retry_attempts=RESUME_RETRY_ATTEMPTS,
                retry_sleep_seconds=RESUME_RETRY_SLEEP_SECONDS,
            )["pod"]
        else:
            payload = build_screening_pod_payload(
                name=self.pod_name,
                gpu_count=1,
                gpu_type_id=self.gpu_type_id,
                template_id=self.template_id,
                container_disk_gb=self.container_disk_gb,
                volume_gb=self.volume_gb,
            )
            pod = ensure_named_screening_pod(
                self.client,
                name=self.pod_name,
                payload=payload,
                retry_attempts=RESUME_RETRY_ATTEMPTS,
                retry_sleep_seconds=RESUME_RETRY_SLEEP_SECONDS,
            )["pod"]

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
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=10",
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
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=10",
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
        source = str(data.get("source", "cached"))
        if source == "retokenize-docs":
            self._ensure_retokenized_dataset(data)
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

    def _ensure_retokenized_dataset(self, data: dict[str, Any]) -> None:
        dataset_name = str(data["dataset_name"])
        train_shards = int(data.get("train_shards", 80))
        repo_id = str(data.get("repo_id", "willdepueoai/parameter-golf"))
        remote_root = str(data.get("remote_root", "datasets"))
        tokenizer_config = str(data["tokenizer_config"])
        export_root = str(data.get("export_root", "/workspace/scylla_export"))
        hf_home = str(data.get("hf_home", "/workspace/.hf"))
        install_packages = [str(pkg) for pkg in data.get("python_packages", ["tokenmonster"])]
        bootstrap_key = (dataset_name, train_shards, tokenizer_config, export_root)
        if bootstrap_key in self._bootstrapped_variants:
            return

        dataset_dir = REMOTE_ROOT / "data" / "datasets" / dataset_name
        ready_check = (
            f"python3 - <<'PY'\n"
            f"from pathlib import Path\n"
            f"root = Path({str(dataset_dir)!r})\n"
            f"train = sorted(root.glob('fineweb_train_*.bin'))\n"
            f"val = sorted(root.glob('fineweb_val_*.bin'))\n"
            f"raise SystemExit(0 if len(train) == {train_shards} and len(val) > 0 else 1)\n"
            f"PY"
        )
        try:
            self._run_ssh_with_retry(self._ssh_shell(f"cd {shlex.quote(str(REMOTE_ROOT))} && {ready_check}"), attempts=1)
            self._bootstrapped_variants.add(bootstrap_key)
            return
        except RunpodError:
            pass

        package_check = " ".join(shlex.quote(pkg) for pkg in install_packages)
        missing_check = (
            "python3 -c "
            + shlex.quote(
                "import importlib.util; "
                f"missing = [name for name in {install_packages!r} if importlib.util.find_spec(name) is None]; "
                "raise SystemExit(0 if not missing else 1)"
            )
        )
        install_cmd = (
            f"{missing_check} || python3 -m pip install --break-system-packages {package_check}"
        )
        symlink_train_cmd = (
            "python3 -c "
            + shlex.quote(
                "from pathlib import Path; "
                f"src = Path({(export_root + '/datasets/' + dataset_name)!r}); "
                f"dst = Path({str(dataset_dir)!r}); "
                "files = sorted(src.glob('fineweb_train_*.bin')); "
                f"limit = {train_shards}; "
                "assert len(files) >= limit, f'expected at least {limit} train shards, found {len(files)}'; "
                "dst.mkdir(parents=True, exist_ok=True); "
                "[(target.unlink() if target.exists() or target.is_symlink() else None, target.symlink_to(path)) "
                " for path in files[:limit] for target in [dst / path.name]]"
            )
        )
        export_cmd = (
            f"cd {shlex.quote(str(REMOTE_ROOT))} && "
            f"mkdir -p {shlex.quote(hf_home)} {shlex.quote(hf_home + '/hub')} && "
            f"{install_cmd} && "
            f"rm -rf {shlex.quote(export_root)} && "
            f"HF_HOME={shlex.quote(hf_home)} HUGGINGFACE_HUB_CACHE={shlex.quote(hf_home + '/hub')} HF_HUB_DISABLE_XET=1 "
            f"python3 data/download_hf_docs_and_tokenize.py "
            f"--repo-id {shlex.quote(repo_id)} "
            f"--remote-root {shlex.quote(remote_root)} "
            f"--output-root {shlex.quote(export_root)} "
            f"--max-train-shards {train_shards} "
            f"--tokenizer-config {shlex.quote(tokenizer_config)} && "
            f"rm -rf {shlex.quote(str(dataset_dir))} && "
            f"mkdir -p {shlex.quote(str(dataset_dir))} && "
            f"for f in {shlex.quote(export_root)}/datasets/{shlex.quote(dataset_name)}/fineweb_val_*.bin; do "
            f"ln -sfn \"$f\" {shlex.quote(str(dataset_dir))}/$(basename \"$f\"); "
            f"done && "
            f"{symlink_train_cmd}"
        )
        self._run_ssh_with_retry(self._ssh_shell(export_cmd), attempts=1)
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
