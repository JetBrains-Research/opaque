"""Real local-process smoke coverage for MLX collectives."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")


def _free_port_range(size: int) -> int:
    for _ in range(20):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        sockets = []
        try:
            for offset in range(size):
                candidate = socket.socket()
                candidate.bind(("127.0.0.1", port + offset))
                sockets.append(candidate)
        except OSError:
            continue
        finally:
            for candidate in sockets:
                candidate.close()
        return port
    raise RuntimeError("Could not reserve a local port range for MLX.")


@pytest.mark.slow
def test_two_process_distributed_profile() -> None:
    launcher = shutil.which("mlx.launch")
    if launcher is None or not mx.distributed.is_available():
        pytest.skip("MLX local distributed launch is unavailable")
    worker = Path(__file__).with_name("_distributed_smoke.py")
    result = subprocess.run(
        [
            launcher,
            "-n",
            "2",
            "--backend",
            "ring",
            "--starting-port",
            str(_free_port_range(2)),
            sys.executable,
            str(worker),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, (
        f"mlx.launch exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
