"""Run tool code in a separate, resource-limited, killable process.

This is the in-process layer only. In production, run the whole model host
inside an outer boundary as well: a gVisor/Firecracker VM or container with no
network egress except an allow-listed proxy, read-only root filesystem,
seccomp, and no credentials mounted. See docs/SAFETY.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:
    import resource
except ImportError:  # non-POSIX
    resource = None  # type: ignore[assignment]


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _limits(cpu_s: int, mem_bytes: int, max_file: int):
    def apply() -> None:
        os.setsid()  # own process group so we can kill descendants
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file, max_file))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run_python(
    code: str,
    timeout: float = 5.0,
    cpu_seconds: int = 5,
    memory_bytes: int = 512 * 1024 * 1024,
    max_file_bytes: int = 1024 * 1024,
) -> SandboxResult:
    with tempfile.TemporaryDirectory(prefix="guardian-") as tmp:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", code],
            cwd=tmp,
            env={"PATH": "/usr/bin:/bin", "HOME": tmp},  # no inherited secrets
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_limits(cpu_seconds, memory_bytes, max_file_bytes),
        )
        try:
            out, err = proc.communicate(timeout=timeout)
            return SandboxResult(proc.returncode, out, err, False)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
            out, err = proc.communicate()
            return SandboxResult(-9, out, err, True)
