"""Execute every code cell of each notebook in order, in one namespace.

Stands in for a real jupyter run (nbconvert is not installed) and is worth keeping:
it makes the notebooks testable in CI without a kernel.
"""
import json, os, sys, time, traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

failed = False
HOME = Path.cwd()
for nb_path in sorted(Path("ipynb").glob("*.ipynb")):
    os.chdir(HOME)          # a notebook that chdirs must not strand the next one
    cells = [c for c in json.loads(nb_path.read_text())["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    print(f"\n=== {nb_path.name}: {len(cells)} code cells ===")
    t0 = time.time()
    for i, cell in enumerate(cells, 1):
        src = cell["source"]
        src = src if isinstance(src, str) else "".join(src)
        try:
            exec(compile(src, f"{nb_path.name}[cell {i}]", "exec"), ns)
        except Exception:
            failed = True
            print(f"  cell {i} FAILED")
            traceback.print_exc(limit=3)
            break
        print(f"  cell {i}/{len(cells)} ok", end="\r")
    else:
        print(f"  all {len(cells)} cells ran in {time.time()-t0:.1f}s")

sys.exit(1 if failed else 0)
