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
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    return apply


def run_python(
    code: str,
    timeout: float = 5.0,
    cpu_seconds: int = 5,
    memory_bytes: int = 512 * 1024 * 1024,
    max_file_bytes: int = 1024 * 1024,
) -> SandboxResult:
    # Output goes to files (not pipes) so RLIMIT_FSIZE also caps how much the
    # child can make the host buffer.
    with tempfile.TemporaryDirectory(prefix="guardian-") as tmp, \
            open(os.path.join(tmp, ".out"), "w+b") as out, \
            open(os.path.join(tmp, ".err"), "w+b") as err:
        work = os.path.join(tmp, "work")
        os.mkdir(work, 0o700)
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", code],
            cwd=work,
            env={"PATH": "/usr/bin:/bin", "HOME": work},  # no inherited secrets
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            close_fds=True,
            preexec_fn=_limits(cpu_seconds, memory_bytes, max_file_bytes),
        )
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(proc.pid, 9)  # also reaps any grandchildren
            except ProcessLookupError:
                pass
            proc.wait()

        def read(f) -> str:
            f.seek(0)
            return f.read(max_file_bytes).decode("utf-8", "replace")

        return SandboxResult(-9 if timed_out else proc.returncode,
                             read(out), read(err), timed_out)
