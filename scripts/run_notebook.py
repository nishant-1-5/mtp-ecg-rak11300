"""Execute notebooks headless, in place, for a given lead set.

    python scripts/run_notebook.py --lead-set lead1 notebooks/10_classical_xgb_rf.ipynb [...]

Outputs are written back into the notebook, so the committed notebook shows the
run that produced the artifacts in models/<lead set>/ and results/<lead set>/.
Requires: pip install nbclient nbformat ipykernel
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-set", choices=["lead1", "3lead"], required=True)
    ap.add_argument("notebooks", nargs="+", type=Path)
    args = ap.parse_args()

    os.environ["MTP_LEAD_SET"] = args.lead_set  # inherited by the kernel
    for path in args.notebooks:
        nb = nbformat.read(path, as_version=4)
        t0 = time.time()
        print(f"[{args.lead_set}] running {path} ...", flush=True)
        client = NotebookClient(nb, timeout=None, kernel_name="python3",
                                resources={"metadata": {"path": str(path.parent)}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, path)  # keep partial outputs for debugging
        print(f"[{args.lead_set}] done {path} in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
