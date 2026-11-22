"""Run the wildlife re-ID MiewID study on Kaggle with one command:
  python scripts/run_on_kaggle.py 01 02   # data check + validation & adapter training
  python scripts/run_on_kaggle.py 03      # locked held-out test set (run ONCE after checking 02)
"""
import json, os, re, shutil, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = "wildlifedatasets/wildlifereid-10k"
STEPS = {
    "01": ("notebooks/01_data_check.ipynb", "miewid-reid-01-data-check", [], False),
    "02": ("notebooks/02_val_and_factors.ipynb", "miewid-reid-02-val-factors", ["01"], True),
    "03": ("notebooks/03_test_set.ipynb", "miewid-reid-03-test-set", ["01", "02"], True),
}
SHOW = ["data_check_summary.json", "val_summary.json", "test_summary.json"]
POLL_SECONDS = 30
MAX_HOURS = 3

def kaggle(*args):
    exe = shutil.which("kaggle")
    cmd = [exe, *args] if exe else [sys.executable, "-m", "kaggle", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout

def username():
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    out = kaggle("kernels", "list", "--mine", "--csv")
    for line in out.splitlines()[1:]:
        if "/" in line:
            return line.split(",")[0].split("/")[0].strip()
    sys.exit("Kaggle username not found via CLI.")

def push(user, step):
    notebook, slug, needs, gpu = STEPS[step]
    sources = [f"{user}/{STEPS[n][1]}" for n in needs]
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(REPO / notebook, Path(tmp) / Path(notebook).name)
        meta = {
            "id": f"{user}/{slug}", "title": slug, "code_file": Path(notebook).name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": sources,
        }
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        print(f"[{step}] {kaggle('kernels', 'push', '-p', tmp).strip()}")
    return f"{user}/{slug}"

def wait(step, kernel_id):
    start = time.time()
    while True:
        out = kaggle("kernels", "status", kernel_id)
        m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', out)
        status = (m.group(1) if m else out.strip()).lower()
        print(f"[{step}] {datetime.now():%H:%M:%S}  {status}", flush=True)
        if "complete" in status:
            return True
        if "error" in status or "cancel" in status:
            print(f"[{step}] stopped with status {status}; log: https://www.kaggle.com/code/{kernel_id}")
            return False
        if time.time() - start > MAX_HOURS * 3600:
            sys.exit(f"[{step}] timeout after {MAX_HOURS}h")
        time.sleep(POLL_SECONDS)

def show_log(target, n=40):
    for log in sorted(target.glob("*.log")):
        raw = log.read_text()
        try: text = "".join(e.get("data", "") for e in json.loads(raw))
        except ValueError: text = raw
        lines = [l for l in text.splitlines() if l.strip()]
        print(f"===== end of {log.name}")
        print("\n".join(lines[-n:]))

def main():
    steps = sys.argv[1:] or ["01", "02"]
    for s in steps:
        if s not in STEPS: sys.exit(f"Unknown step {s}")
    user = username()
    run_dir = REPO / "runs" / f"{datetime.now():%Y-%m-%d_%H-%M}"
    for step in steps:
        kernel_id = push(user, step)
        time.sleep(10)
        ok = wait(step, kernel_id)
        target = run_dir / step
        target.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", kernel_id, "-p", str(target))
        if not ok:
            show_log(target)
            sys.exit(f"[{step}] failed; logs in {target}")
        print(f"[{step}] output in {target}")
    for name in SHOW:
        for f in sorted(run_dir.rglob(name)):
            print(f"===== {f.name}")
            print(f.read_text()[:4000])

if __name__ == "__main__":
    main()
