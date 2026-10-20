"""Fail-closed Gate 1; synthetic canaries never inspect actual game files."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.sandbox import run_sandbox

FORBIDDEN_TOKENS = ['environment_files', 'ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59', 'lf52', 'lp85', 'ls20', 'm0r0', 'r11l', 're86', 's5i5', 'sb26', 'sc25', 'sk48', 'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30']


def audit(root: Path = ROOT, *, loop_timeout: float = 15.0) -> dict:
    violations: list[str] = []
    checks: dict[str, bool] = {}
    evidence: dict = {}
    sources = sorted((root / "agent").rglob("*.py"))
    checks["deployed_sources_present"] = all(
        (root / "agent" / name).is_file() for name in ("my_agent.py", "arclog.py"))
    for path in sources:
        content = path.read_text(encoding="utf-8").lower()
        tokens = [t for t in FORBIDDEN_TOKENS if t in content]
        # Reject any short game identifier, including identifiers not in the list.
        tokens += re.findall(r"(?<![a-z0-9_])[a-z]{2}\d{2}(?:-[a-f0-9]+)?(?![a-z0-9_])", content)
        if tokens:
            violations.append(f"{path.relative_to(root)}: forbidden tokens {sorted(set(tokens))}")
    sandbox = root / ".venv_agent_sandbox" / "bin" / "python"
    checks["sandbox_present"] = sandbox.is_file()
    checks["engine_import_denied"] = False
    if sandbox.is_file():
        # A clean venv must fail specifically with ModuleNotFoundError. Any other
        # subprocess error (including a broken interpreter) fails closed.
        probe = """import sys
try:
    import arc_agi
except ModuleNotFoundError as error:
    if error.name == 'arc_agi':
        print('ENGINE_ABSENT')
        sys.exit(0)
sys.exit(1)
"""
        try:
            result = subprocess.run([str(sandbox), "-I", "-c", probe],
                                    check=False, capture_output=True, text=True, timeout=15,
                                    env={}, cwd=root)
            checks["engine_import_denied"] = (
                result.returncode == 0 and result.stdout == "ENGINE_ABSENT\n")
            evidence["engine_probe_returncode"] = result.returncode
            with tempfile.TemporaryDirectory(prefix="containment-") as temp:
                fixture = Path(temp)
                forbidden = fixture / "environment_files"
                forbidden.mkdir()
                canary = forbidden / "canary.txt"
                canary.write_text("synthetic-secret")
                scratch = fixture / "scratch"
                scratch.mkdir()
                (scratch / "linked").symlink_to(forbidden, target_is_directory=True)
                attempts = {
                    "absolute_read": f"open({str(canary)!r}).read()",
                    "traversal_read": "open('../environment_files/canary.txt').read()",
                    "symlink_read": "open('linked/canary.txt').read()",
                    "write": f"open({str(canary)!r}, 'w').write('changed')",
                    "symlink_write": "open('linked/new.txt', 'w').write('new')",
                    "directory_list": f"__import__('os').listdir({str(forbidden)!r})",
                    "subprocess_escape": "__import__('os').system('cat ../environment_files/canary.txt')",
                    "fork_escape": "__import__('os').fork()",
                    "exec_escape": "__import__('os').execv('/bin/true', ['/bin/true'])",
                    "native_open": f"__import__('ctypes').CDLL(None).open({str(canary).encode()!r}, 0)",
                }
                # Assert denial in-worker; a crash/initialization failure cannot pass.
                for name, expression in attempts.items():
                    denied_results = (-1, 127 << 8) if name == "subprocess_escape" else (-1,)
                    source = f"""try:
    result = {expression}
except PermissionError:
    print('DENIED')
else:
    assert isinstance(result, int) and result in {denied_results!r}, 'access succeeded'
    print('DENIED')
"""
                    run = run_sandbox(source, python=sandbox, cwd=scratch,
                                      controlled_fixture=True)
                    checks[name] = run["ok"] and run["stdout"] == "DENIED\n"
                    evidence[name] = run
                checks["canary_unchanged"] = canary.read_text() == "synthetic-secret"
                checks["write_not_created"] = not (forbidden / "new.txt").exists()
                run = run_sandbox("print(sum(range(10)))", python=sandbox, cwd=scratch,
                                  controlled_fixture=True)
                checks["compute_works"] = run["ok"] and run["stdout"] == "45\n"
                if os.getenv("ENABLE_SANDBOX_LIMITS", "1") == "1":
                    source = "try:\n    bytearray(600 * 1024**2)\nexcept MemoryError:\n    print('MEMORY_CAPPED')"
                    run = run_sandbox(source, python=sandbox, cwd=scratch,
                                      controlled_fixture=True)
                    checks["memory_capped"] = run["ok"] and run["stdout"] == "MEMORY_CAPPED\n"
                    evidence["memory"] = run
                run = run_sandbox("raise ValueError('planner-canary')", python=sandbox,
                                  cwd=scratch, controlled_fixture=True)
                checks["planner_failure_visible"] = (run["ready"] and not run["ok"]
                                                     and "planner-canary" in run["stderr"])
                run = run_sandbox("while True: pass", python=sandbox, cwd=scratch,
                                  timeout=loop_timeout, controlled_fixture=True)
                checks["loop_terminated"] = (run["ready"] and not run["ok"] and run["timeout"]
                                             and run["elapsed_seconds"] <= loop_timeout + 2)
                evidence["loop"] = run
                # Missing interpreter must never produce a successful worker.
                run = run_sandbox("print(1)", python=fixture / "missing", cwd=scratch)
                checks["missing_worker_fails"] = not run["ok"] and not run["ready"]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            violations.append(f"sandbox probe failed: {type(error).__name__}: {error}")
    for name, passed in checks.items():
        if not passed:
            violations.append(f"failed check: {name}")
    return {"containment_passed": not violations and all(checks.values()),
            "sandbox_isolated": all(checks.values()), "checks": checks,
            "violations": violations, "evidence": evidence,
            "resource_limits_enabled": os.getenv("ENABLE_SANDBOX_LIMITS", "1") == "1"}


def main() -> int:
    try:
        report = audit()
    except Exception as error:  # noqa: BLE001 - audit exceptions must produce a failed report
        report = {"containment_passed": False, "sandbox_isolated": False,
                  "violations": [f"audit failed: {type(error).__name__}: {error}"]}
    (ROOT / "containment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["containment_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
