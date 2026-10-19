"""Audits agent/ for engine/filesystem leakage and writes containment.json."""
import json
import pathlib
import subprocess
import sys

FORBIDDEN_TOKENS = [
    "environment_files",
    "ls20",
    "ft09",
    "vc33",
    "re86",
]

def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    agent_dir = root / "agent"
    violations = []

    for py_file in agent_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in content:
                violations.append(f"{py_file.name}: contains forbidden token '{token}'")

    sandbox_py = root / ".venv_agent_sandbox" / "bin" / "python"
    sandbox_isolated = True
    if sandbox_py.exists():
        proc = subprocess.run(
            [str(sandbox_py), "-c", "import arc_agi"],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            sandbox_isolated = False
            violations.append(".venv_agent_sandbox can import arc_agi engine internals")

    passed = len(violations) == 0 and sandbox_isolated
    report = {
        "containment_passed": passed,
        "sandbox_isolated": sandbox_isolated,
        "violations": violations,
    }
    (root / "containment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())