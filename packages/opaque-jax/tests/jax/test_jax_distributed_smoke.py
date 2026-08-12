"""Real local-process smoke coverage for JAX collectives."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.slow
def test_two_process_distributed_profile() -> None:
    worker = Path(__file__).with_name("_distributed_smoke.py")
    coordinator = f"127.0.0.1:{_free_port()}"
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "no_proxy"}
    }
    env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(worker),
                "--coordinator",
                coordinator,
                "--rank",
                str(rank),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    outputs = []
    try:
        outputs.extend(process.communicate(timeout=60) for process in processes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)

    failures = [
        f"rank {rank} exited {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        for rank, (process, (stdout, stderr)) in enumerate(
            zip(processes, outputs, strict=True)
        )
        if process.returncode != 0
    ]
    if failures and all(
        "Unable to find address for:" in stderr for _, stderr in outputs
    ):
        pytest.skip("JAX Gloo cannot resolve this host for local process collectives")
    assert not failures, "\n\n".join(failures)
