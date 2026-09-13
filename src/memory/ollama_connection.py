"""Unified Ollama connection for ALLI.

Handles automatic endpoint discovery across Windows PowerShell, WSL Ubuntu,
and VS Code terminals. Creates ONE shared ollama.Client used throughout
the application.

Priority for endpoint selection:
  1. OLLAMA_HOST environment variable (explicit override)
  2. http://127.0.0.1:11434  (native Windows, or WSL if Ollama runs in WSL)
  3. http://<gateway>:11434   (WSL2 → Windows host via default gateway)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

import ollama


# ---------------------------------------------------------------------------
# WSL detection
# ---------------------------------------------------------------------------

def _is_wsl() -> bool:
    """Detect whether we are running inside WSL (Windows Subsystem for Linux)."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", "r") as f:
            version = f.read().lower()
            return "microsoft" in version or "wsl" in version
    except OSError:
        return False


def _get_wsl_gateway_ip() -> Optional[str]:
    """Extract the Windows host IP from the WSL2 default gateway.

    This is the IP that WSL2 uses to reach the Windows host network.
    It changes after each WSL restart but is always the default gateway.
    """
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Output: "default via 172.25.80.1 dev eth0 ..."
        parts = result.stdout.strip().split()
        if "via" in parts:
            idx = parts.index("via")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Endpoint discovery
# ---------------------------------------------------------------------------

def _try_endpoint(url: str, timeout: float = 2.0) -> bool:
    """Probe whether Ollama is reachable at the given URL."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def discover_ollama_endpoint() -> str:
    """Determine the correct Ollama endpoint for this environment.

    Returns the first reachable URL, or a best-guess default.
    """
    localhost = "http://127.0.0.1:11434"

    # 1. Explicit override — validated, not blindly trusted.
    #    OLLAMA_HOST is often set to a *server listen* address (e.g. "0.0.0.0:11434"),
    #    which is not a valid client connect target. Normalize it and probe: if the
    #    override is reachable, use it; otherwise fall through to discovery.
    explicit = os.environ.get("OLLAMA_HOST", "").strip()
    if explicit:
        candidate = explicit
        # Ensure a scheme so ollama.Client parses it correctly
        if "://" not in candidate:
            candidate = "http://" + candidate
        # A server bind address of 0.0.0.0 means "all interfaces" — as a client
        # we connect via loopback instead.
        if "://0.0.0.0" in candidate:
            candidate = candidate.replace("://0.0.0.0", "://" + "127.0.0.1", 1)
        if _try_endpoint(candidate):
            return candidate
        # Unreachable override — try loopback before giving up on it
        if candidate != localhost and _try_endpoint(localhost):
            return localhost

    # 2. Try localhost (works on native Windows, or if Ollama runs inside WSL)
    if _try_endpoint(localhost):
        return localhost

    # 3. If in WSL, try the Windows host gateway IP
    if _is_wsl():
        gw = _get_wsl_gateway_ip()
        if gw:
            gw_url = f"http://{gw}:11434"
            if _try_endpoint(gw_url):
                return gw_url

    # 4. Nothing reachable — return localhost as default (health check will report failure)
    return localhost


# ---------------------------------------------------------------------------
# Health check with retry
# ---------------------------------------------------------------------------

def _socket_check(host: str, port: int, timeout: float = 1.5) -> bool:
    """Quick TCP socket check to see if host:port is listening."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def _extract_host_port(url: str) -> tuple[str, int]:
    """Extract host and port from a URL like http://host:port."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    return host, port


def check_ollama_health(
    client: ollama.Client,
    endpoint: str,
    *,
    retries: int = 3,
    delay: float = 2.0,
) -> tuple[bool, str]:
    """Check if Ollama is reachable, retrying briefly for startup delays.

    Returns (ok, message) where message describes the result.
    """
    host, port = _extract_host_port(endpoint)

    # Quick socket-level pre-check to avoid long httpx timeouts on unreachable hosts
    if not _socket_check(host, port, timeout=1.5):
        return False, (
            f"Ollama unreachable at {endpoint} — host:{port} not accepting connections.\n"
            f"Ensure Ollama is installed and running."
        )

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            listed = client.list()
            models = getattr(listed, "models", None)
            if models is None and isinstance(listed, dict):
                models = listed.get("models", [])
            return True, f"Ollama reachable at {endpoint}"
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(delay)

    return False, (
        f"Ollama unreachable at {endpoint} after {retries} attempts.\n"
        f"Last error: {last_error}\n"
        f"Ensure Ollama is installed and running."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_ollama_client() -> tuple[ollama.Client, str]:
    """Create a single shared ollama.Client with auto-discovered endpoint.

    Returns (client, endpoint_url).
    """
    endpoint = discover_ollama_endpoint()
    client = ollama.Client(host=endpoint)
    return client, endpoint
