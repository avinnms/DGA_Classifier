import argparse
import csv
import sys
import subprocess
import random
from datetime import date, timedelta
from pathlib import Path

def dated(flag="-d", fmt="%Y-%m-%d"):
    return lambda dates: [[flag, d.strftime(fmt)] for d in dates]


RANDOM = {
    "necurs": dated(),
    "qakbot": dated(),
    "pushdo": dated(),
    "chinad": dated(),
    "tufik": dated(),
    "nymaim2": dated(),
    "newgoz": dated(),
    "corebot": dated(),
    "suppobox": lambda dates: [[str(s), "-t", d.strftime("%Y-%m-%d 00:00:00")] for d in dates for s in (1,2,3)],
    "simda": lambda dates: [[]],
    "banjori": lambda dates: [[]],
    "tinba": lambda dates: [[]],
    "ramnit": lambda dates: [["-k", "-x", "50"] for _ in range(20)],
}

def run(repo: Path, family: str, arg_list, per_family: int):
    script = repo / family / "dga.py"
    seen = set()
    for args in arg_list:
        result = subprocess.run(
            [sys.executable, "dga.py", *args], cwd=script.parent, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0: 
            print(f" [warn] {family} {args} failed: {result.stderr.strip().splitlines()[-1:]}")
            continue
        for line in result.stdout.splitlines():
            d = line.strip().lower()
            if d and "." in d and " " not in d:
                seen.add(d)
        if len(seen) >= per_family:
            break
    domains = sorted(seen)
    random.Random(42).shuffle(domains)
    return domains[:per_family]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--out", default="dga_domains.csv")
    p.add_argument("--per-family", type=int, default=2000, help="cap per family so that no single family can dominate")
    p.add_argument("--days", type=int, default=60, help="how many dates to sample now")
    p.add_argument("--families", nargs="*", default=list(RANDOM))
    a = p.parse_args()

    a.repo = a.repo.expanduser().resolve()
    if not (a.repo / "necurs" / "dga.py").exists():
        sys.exit(f"{a.repo} doesn't look like the domain_generation_algorithms repo -- check --repo")

    dates = [date(2026, 1, 1) + timedelta(days=7 * i) for i in range(a.days)]

    rows = []
    for fam in a.families:
        if fam not in RANDOM:
            print(f" [skip] no recipe for {fam}")
            continue
        domains = run(a.repo, fam, RANDOM[fam](dates), a.per_family)
        print(f"{fam:12s} {len(domains):6d} domains")
        rows += [(d, fam) for d in domains]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "family"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
