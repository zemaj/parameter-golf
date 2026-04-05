from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, Optional

from runpod_client import RunpodClient, RunpodError
from runpod_lifecycle import build_screening_pod_payload, ensure_named_screening_pod
from runpod_config import (
    DEFAULT_CONTAINER_DISK_GB,
    DEFAULT_SCREENING_GPU_TYPE_ID,
    DEFAULT_SCREENING_POD_NAME,
    DEFAULT_SCREENING_TEMPLATE_ID,
    DEFAULT_VOLUME_GB,
)


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _find_pod_by_name(pods: Iterable[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    matches = [pod for pod in pods if pod.get("name") == name]
    if not matches:
        return None
    if len(matches) > 1:
        ids = ", ".join(pod["id"] for pod in matches)
        raise RunpodError(f"Multiple pods found with name {name!r}: {ids}")
    return matches[0]


def _build_ssh_command(pod: Dict[str, Any], identity_file: str) -> str:
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    ssh_port = port_mappings.get("22") or port_mappings.get(22)
    if public_ip and ssh_port:
        return f"ssh root@{public_ip} -p {ssh_port} -i {identity_file}"
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if str(port.get("privatePort")) == "22" and port.get("publicPort") and port.get("ip"):
            return f"ssh root@{port['ip']} -p {port['publicPort']} -i {identity_file}"
    raise RunpodError("Pod does not have a public SSH port yet. Wait a bit and re-run the ssh command.")


def cmd_list_pods(client: RunpodClient, _: argparse.Namespace) -> None:
    _print_json(list(client.list_pods()))


def cmd_list_gpus(client: RunpodClient, args: argparse.Namespace) -> None:
    gpu_types = list(client.list_gpu_types())
    if args.match:
        needle = args.match.lower()
        gpu_types = [
            gpu for gpu in gpu_types
            if needle in str(gpu.get("id", "")).lower() or needle in str(gpu.get("displayName", "")).lower()
        ]
    _print_json(gpu_types)


def cmd_status(client: RunpodClient, args: argparse.Namespace) -> None:
    _print_json(client.get_pod(args.pod_id))


def cmd_stop(client: RunpodClient, args: argparse.Namespace) -> None:
    pod_id = args.pod_id
    if not pod_id:
        pod = _find_pod_by_name(client.list_pods(), args.name)
        if pod is None:
            raise RunpodError(f"No pod found with name {args.name!r}")
        pod_id = pod["id"]
    _print_json(client.stop_pod(pod_id))


def cmd_resume(client: RunpodClient, args: argparse.Namespace) -> None:
    _print_json(client.resume_pod(args.pod_id))


def cmd_delete(client: RunpodClient, args: argparse.Namespace) -> None:
    pod_id = args.pod_id
    if not pod_id:
        pod = _find_pod_by_name(client.list_pods(), args.name)
        if pod is None:
            raise RunpodError(f"No pod found with name {args.name!r}")
        pod_id = pod["id"]
    _print_json(client.delete_pod(pod_id))


def cmd_ssh(client: RunpodClient, args: argparse.Namespace) -> None:
    _print_json({"pod_id": args.pod_id, "ssh": _build_ssh_command(client.get_pod(args.pod_id), args.identity_file)})


def cmd_up_screening(client: RunpodClient, args: argparse.Namespace) -> None:
    payload = build_screening_pod_payload(
        name=args.name,
        gpu_count=args.gpu_count,
        gpu_type_id=args.gpu_type_id,
        template_id=args.template_id,
        container_disk_gb=args.container_disk_gb,
        volume_gb=args.volume_gb,
    )
    _print_json(ensure_named_screening_pod(client, name=args.name, payload=payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runpod helpers for Parameter Golf screening pods.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_pods = subparsers.add_parser("list-pods", help="List all pods in the current Runpod account.")
    list_pods.set_defaults(func=cmd_list_pods)

    list_gpus = subparsers.add_parser("list-gpus", help="List available GPU types and pricing metadata.")
    list_gpus.add_argument("--match", help="Filter GPU names by substring, e.g. H100.")
    list_gpus.set_defaults(func=cmd_list_gpus)

    status = subparsers.add_parser("status", help="Get a pod by id.")
    status.add_argument("pod_id")
    status.set_defaults(func=cmd_status)

    stop = subparsers.add_parser("stop", help="Stop a pod by id or by name.")
    stop.add_argument("pod_id", nargs="?")
    stop.add_argument("--name", default=DEFAULT_SCREENING_POD_NAME)
    stop.set_defaults(func=cmd_stop)

    resume = subparsers.add_parser("resume", help="Resume a stopped pod by id.")
    resume.add_argument("pod_id")
    resume.set_defaults(func=cmd_resume)

    delete = subparsers.add_parser("delete", help="Delete a pod by id or by name.")
    delete.add_argument("pod_id", nargs="?")
    delete.add_argument("--name")
    delete.set_defaults(func=cmd_delete)

    ssh = subparsers.add_parser("ssh", help="Print a full SSH command for a running pod.")
    ssh.add_argument("pod_id")
    ssh.add_argument("--identity-file", default="~/.ssh/id_rsa")
    ssh.set_defaults(func=cmd_ssh)

    up = subparsers.add_parser(
        "up-screening",
        help="Resume an existing screening pod by name, or create one if it does not exist.",
    )
    up.add_argument("--name", default=DEFAULT_SCREENING_POD_NAME)
    up.add_argument("--gpu-count", type=int, default=1)
    up.add_argument("--gpu-type-id", default=DEFAULT_SCREENING_GPU_TYPE_ID)
    up.add_argument("--template-id", default=DEFAULT_SCREENING_TEMPLATE_ID)
    up.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    up.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    up.set_defaults(func=cmd_up_screening)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        client = RunpodClient.from_env()
        args.func(client, args)
    except RunpodError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
