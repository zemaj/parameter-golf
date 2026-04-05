from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from runpod_client import RunpodClient, RunpodError
from runpod_config import (
    DEFAULT_CONTAINER_DISK_GB,
    DEFAULT_PORTS,
    DEFAULT_SCREENING_GPU_TYPE_ID,
    DEFAULT_SCREENING_TEMPLATE_ID,
    DEFAULT_VOLUME_GB,
    DEFAULT_VOLUME_MOUNT_PATH,
)


RESUME_RETRY_SLEEP_SECONDS = 15
RESUME_RETRY_ATTEMPTS = 12
RETRYABLE_CAPACITY_SNIPPETS = (
    "not enough free gpus",
    "not enough free gpus on the host machine",
    "no instances currently available",
)


class RunpodCapacityRetryError(RunpodError):
    pass


def is_retryable_capacity_error(exc: RunpodError) -> bool:
    message = str(exc).lower()
    return any(snippet in message for snippet in RETRYABLE_CAPACITY_SNIPPETS)


def build_screening_pod_payload(
    *,
    name: str,
    gpu_count: int = 1,
    gpu_type_id: str = DEFAULT_SCREENING_GPU_TYPE_ID,
    template_id: str = DEFAULT_SCREENING_TEMPLATE_ID,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
) -> dict[str, Any]:
    return {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": container_disk_gb,
        "gpuCount": gpu_count,
        "gpuTypeIds": [gpu_type_id],
        "interruptible": False,
        "name": name,
        "ports": list(DEFAULT_PORTS),
        "supportPublicIp": True,
        "templateId": template_id,
        "volumeInGb": volume_gb,
        "volumeMountPath": DEFAULT_VOLUME_MOUNT_PATH,
    }


def pods_with_name(pods: Iterable[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [pod for pod in pods if pod.get("name") == name]


def _preferred_pod(pods: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not pods:
        return None
    running = [pod for pod in pods if pod.get("desiredStatus") == "RUNNING"]
    if len(running) > 1:
        ids = ", ".join(pod["id"] for pod in running)
        raise RunpodError(f"Multiple running pods found with the same name: {ids}")
    if running:
        return running[0]
    return max(pods, key=lambda pod: str(pod.get("createdAt", "")))


def _retry_capacity_operation(
    action_name: str,
    action: Callable[[], dict[str, Any]],
    *,
    retry_attempts: int = RESUME_RETRY_ATTEMPTS,
    retry_sleep_seconds: int = RESUME_RETRY_SLEEP_SECONDS,
) -> dict[str, Any]:
    last_error: RunpodError | None = None
    for attempt in range(retry_attempts):
        try:
            return action()
        except RunpodError as exc:
            if not is_retryable_capacity_error(exc):
                raise
            last_error = exc
            if attempt + 1 == retry_attempts:
                break
            time.sleep(retry_sleep_seconds)
    raise RunpodCapacityRetryError(
        f"Runpod could not {action_name} after repeated capacity retries. Last error: {last_error}"
    )


def create_pod_with_retries(
    client: RunpodClient,
    payload: dict[str, Any],
    *,
    retry_attempts: int = RESUME_RETRY_ATTEMPTS,
    retry_sleep_seconds: int = RESUME_RETRY_SLEEP_SECONDS,
) -> dict[str, Any]:
    return _retry_capacity_operation(
        "allocate a screening pod",
        lambda: client.create_pod(payload),
        retry_attempts=retry_attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )


def resume_pod_with_retries(
    client: RunpodClient,
    pod_id: str,
    *,
    retry_attempts: int = RESUME_RETRY_ATTEMPTS,
    retry_sleep_seconds: int = RESUME_RETRY_SLEEP_SECONDS,
) -> dict[str, Any]:
    return _retry_capacity_operation(
        f"resume screening pod {pod_id}",
        lambda: client.resume_pod(pod_id),
        retry_attempts=retry_attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )


def ensure_named_screening_pod(
    client: RunpodClient,
    *,
    name: str,
    payload: dict[str, Any],
    retry_attempts: int = RESUME_RETRY_ATTEMPTS,
    retry_sleep_seconds: int = RESUME_RETRY_SLEEP_SECONDS,
) -> dict[str, Any]:
    pods = pods_with_name(client.list_pods(), name)
    preferred = _preferred_pod(pods)
    deleted_pod_ids: list[str] = []

    if preferred is not None:
        for pod in pods:
            if pod["id"] == preferred["id"]:
                continue
            client.delete_pod(pod["id"])
            deleted_pod_ids.append(pod["id"])

        if preferred.get("desiredStatus") == "RUNNING":
            return {"action": "already_running", "pod": preferred, "deleted_pod_ids": deleted_pod_ids}

        try:
            resumed = resume_pod_with_retries(
                client,
                preferred["id"],
                retry_attempts=retry_attempts,
                retry_sleep_seconds=retry_sleep_seconds,
            )
            return {"action": "resumed", "pod": resumed, "deleted_pod_ids": deleted_pod_ids}
        except RunpodCapacityRetryError:
            client.delete_pod(preferred["id"])
            deleted_pod_ids.append(preferred["id"])

    created = create_pod_with_retries(
        client,
        payload,
        retry_attempts=retry_attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    action = "recreated" if preferred is not None else "created"
    return {"action": action, "pod": created, "deleted_pod_ids": deleted_pod_ids}


def ensure_screening_pod_by_id(
    client: RunpodClient,
    *,
    pod_id: str,
    retry_attempts: int = RESUME_RETRY_ATTEMPTS,
    retry_sleep_seconds: int = RESUME_RETRY_SLEEP_SECONDS,
) -> dict[str, Any]:
    pod = client.get_pod(pod_id)
    if pod.get("desiredStatus") == "RUNNING":
        return {"action": "already_running", "pod": pod, "deleted_pod_ids": []}
    resumed = resume_pod_with_retries(
        client,
        pod_id,
        retry_attempts=retry_attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    return {"action": "resumed", "pod": resumed, "deleted_pod_ids": []}
