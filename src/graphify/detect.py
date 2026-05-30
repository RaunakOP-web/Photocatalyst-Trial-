"""
detect.py
Recursively scans the workspace for files relevant to the photocatalyst project
and groups them by type (code, datasets, documentation, models, notebooks).
"""

import os
import re

# File types we want to capture
CODE_EXTS = {".py"}
NOTEBOOK_EXTS = {".ipynb"}
DATA_EXTS = {".json", ".csv", ".xlsx", ".xlsm"}
DOC_EXTS = {".md", ".pdf", ".txt"}
MODEL_EXTS = {".joblib", ".pkl"}

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".graphify_cache",
    "graphify-out",
    "venv",
    ".venv",
    "env",
    "build",
    "dist",
    "node_modules",
}

def scan_workspace(root_dir="."):
    """
    Recursively scans the directory and categorizes files.
    Returns a dict mapping categories to lists of absolute file paths.
    """
    categorized_files = {
        "code": [],
        "notebooks": [],
        "datasets": [],
        "documents": [],
        "models": [],
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude directories in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for fname in filenames:
            fpath = os.path.abspath(os.path.join(dirpath, fname))
            ext = os.path.splitext(fname)[1].lower()

            # Ignore gitkeep and other empty tracking files
            if fname.startswith(".git") or fname == "Thumbs.db" or fname == ".DS_Store":
                continue

            if ext in CODE_EXTS:
                categorized_files["code"].append(fpath)
            elif ext in NOTEBOOK_EXTS:
                categorized_files["notebooks"].append(fpath)
            elif ext in DATA_EXTS:
                categorized_files["datasets"].append(fpath)
            elif ext in MODEL_EXTS:
                categorized_files["models"].append(fpath)
            elif ext in DOC_EXTS:
                # Exclude temporary graph reports or specific out files if they leak
                if "GRAPH_REPORT" in fname or "walkthrough" in fname or "task.md" in fname or "implementation_plan" in fname or "architecture_comparison" in fname:
                    continue
                categorized_files["documents"].append(fpath)

    # Print summary
    print("=== Graphify File Detection ===")
    for cat, paths in categorized_files.items():
        print(f"  {cat.capitalize()}: {len(paths)} files found")
    
    return categorized_files

if __name__ == "__main__":
    scan_workspace()
