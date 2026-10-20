"""Engine-free compute worker: stdin/stdout only; Linux seccomp is mandatory.

Callers pass immutable evidence as source literals/JSON, never filesystem handles.
No worker filesystem access is allowed, including scratch (host mediates storage).
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

# Runs under the sandbox interpreter, not the engine-enabled host interpreter.
_WORKER = r'''
import collections
import ctypes
import errno
import json
import math
import os
import resource
import sys
import traceback

source = sys.stdin.read()
limited = os.environ.get("ENABLE_SANDBOX_LIMITS", "1") == "1"
if limited:
    for kind, limit in ((resource.RLIMIT_AS, 512 * 1024**2),
                        (resource.RLIMIT_NPROC, 32), (resource.RLIMIT_NOFILE, 32),
                        (resource.RLIMIT_FSIZE, 1024**2), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(kind, (limit, limit))
# The allowlist remains mandatory even in resource ablations. It cannot be
# removed by generated code, including ctypes/native code in the same process.
lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
lib.seccomp_release.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(0x00050000 | errno.EPERM)
if not ctx:
    raise RuntimeError("seccomp initialization failed")
for name in ("read", "write", "close", "fstat", "lseek", "brk", "mmap", "mprotect",
             "munmap", "mremap", "madvise", "futex", "clock_gettime", "gettimeofday",
             "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigaltstack",
             "getpid", "gettid", "getrandom", "sched_yield", "exit", "exit_group"):
    number = lib.seccomp_syscall_resolve_name(name.encode())
    if number < 0 or lib.seccomp_rule_add(ctx, 0x7fff0000, number, 0) != 0:
        raise RuntimeError("seccomp rule failed: " + name)
if lib.seccomp_load(ctx) != 0:
    raise RuntimeError("seccomp load failed")
lib.seccomp_release(ctx)
print("SANDBOX_READY", flush=True)
try:
    exec(compile(source, "<planner>", "exec"), {"__name__": "__main__"})
except BaseException as error:
    # Avoid traceback source lookup, which would itself need file access.
    print(type(error).__name__ + ": " + str(error), file=sys.stderr)
    sys.exit(1)
'''


def run_sandbox(source: str, *, python: Path, cwd: Path,
                timeout: float = 15.0, controlled_fixture: bool = False) -> dict[str, Any]:
    """Execute a planner, report errors explicitly and always reap the worker.

    Resource ablations require a controlled fixture; security is never ablated.
    Optional shorter timeouts support tests; production cannot exceed 15 seconds.
    """
    started = time.monotonic()
    limited = os.getenv("ENABLE_SANDBOX_LIMITS", "1") == "1"
    if not limited and not controlled_fixture:
        raise ValueError("resource ablation requires controlled_fixture=True")
    if not python.is_file():
        return {"ok": False, "error": "missing_sandbox", "ready": False}
    if not 0 < timeout <= 15:
        raise ValueError("timeout must be in (0, 15]")
    env = {"ENABLE_SANDBOX_LIMITS": "1" if limited else "0"}
    # Regular files plus RLIMIT_FSIZE bound output without pipe deadlock.
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(
                [str(python.absolute()), "-I", "-c", _WORKER], stdin=subprocess.PIPE,
                stdout=output, stderr=errors, cwd=cwd, env=env,
                close_fds=True, start_new_session=True,
            )
        except OSError as error:
            return {"ok": False, "ready": False, "error": str(error)}
        timed_out = False
        try:
            process.communicate(source.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # No child creation is permitted; reap the whole group defensively.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        output.seek(0)
        errors.seek(0)
        stdout = output.read(1024**2).decode(errors="replace")
        stderr = errors.read(1024**2).decode(errors="replace")
    ready = stdout.startswith("SANDBOX_READY\n")
    return {"ok": ready and not timed_out and process.returncode == 0,
            "ready": ready, "timeout": timed_out, "returncode": process.returncode,
            "stdout": stdout.removeprefix("SANDBOX_READY\n"), "stderr": stderr,
            "error": "timeout" if timed_out else ("worker_failed" if process.returncode else None),
            "elapsed_seconds": time.monotonic() - started}
