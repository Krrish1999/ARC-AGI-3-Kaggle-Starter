"""Summarizes local run logs into Failure Scorecards for the PhD agents."""
import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--out", default="research_lab/traces/latest_summary.json")
    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    summary = {"games_found": 0, "files": []}
    if log_dir.exists():
        for p in sorted(log_dir.rglob("*")):
            if p.is_file():
                summary["files"].append(str(p))
        summary["games_found"] = len(summary["files"])

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote scorecard summary ({summary['games_found']} files) to {out_path}")

if __name__ == "__main__":
    main()