from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REST_BASE_URL = "https://rest.runpod.io/v1"


class RunpodError(RuntimeError):
    pass


def load_dotenv(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _json_request(url: str, *, method: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RunpodError(f"Runpod API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RunpodError(f"Runpod API network error: {exc.reason}") from exc
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RunpodError(f"Runpod API returned invalid JSON: {raw[:500]}") from exc


class RunpodClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise RunpodError("RUNPOD_API_KEY is missing. Put it in .env or export it in your shell.")
        self.api_key = api_key

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "RunpodClient":
        load_dotenv(env_path)
        return cls(os.environ.get("RUNPOD_API_KEY", ""))

    def rest(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return _json_request(
            f"{REST_BASE_URL}{path}",
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )

    def list_pods(self) -> Iterable[Dict[str, Any]]:
        data = self.rest("GET", "/pods")
        if not isinstance(data, list):
            raise RunpodError(f"Unexpected response from list pods: {data}")
        return data

    def get_pod(self, pod_id: str) -> Dict[str, Any]:
        data = self.rest("GET", f"/pods/{pod_id}")
        if not isinstance(data, dict) or not data.get("id"):
            raise RunpodError(f"Pod not found: {pod_id}")
        return data

    def list_gpu_types(self) -> Iterable[Dict[str, Any]]:
        data = self.rest("GET", "/openapi.json")
        seen = set()
        gpu_ids = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                enum_values = node.get("enum")
                if isinstance(enum_values, list):
                    for value in enum_values:
                        if isinstance(value, str) and (value.startswith("NVIDIA ") or value.startswith("AMD ")):
                            if value not in seen:
                                seen.add(value)
                                gpu_ids.append(value)
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(data)
        return [{"id": gpu_id, "displayName": gpu_id} for gpu_id in gpu_ids]

    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        return self.rest("POST", f"/pods/{pod_id}/stop")

    def resume_pod(self, pod_id: str) -> Dict[str, Any]:
        return self.rest("POST", f"/pods/{pod_id}/start")

    def delete_pod(self, pod_id: str) -> Dict[str, Any]:
        return self.rest("DELETE", f"/pods/{pod_id}")

    def create_pod(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.rest("POST", "/pods", payload)
