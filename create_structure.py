# create_structure.py
# Creates the initial backend folder structure for the Sighnal project.
# Windows base path (default): D:\Hare Krishna_ai_blog

from __future__ import annotations

import argparse
from pathlib import Path


DIRS = [
    # App root
    "app",
    "app/api",
    "app/api/v1",
    "app/core",
    "app/models",
    "app/models/schemas",
    "app/services",
    "app/services/ingestion",
    "app/services/preprocess",
    "app/services/retrieval",
    "app/services/generation",
    "app/services/qc",
    "app/workers",
    "app/workers/langgraph",
    "app/utils",

    # Tests + docs
    "tests",
    "docs",
    "docs/api_contract",
    "docs/erd",
    "docs/day_plan",

    # Operational folders
    "scripts",
    "migrations",
    "storage",
    "storage/tmp",
    "storage/local_cache",

    # Optional infra placeholders (useful later)
    "infra",
    "infra/docker",
    "infra/terraform",
]


def _print_tree(base: Path) -> None:
    """Print a simple directory tree (directories only) for what we created."""
    base = base.resolve()
    print(f"\n📁 Project root: {base}\n")
    created = sorted({(base / d).resolve() for d in DIRS}, key=lambda p: (len(p.parts), str(p).lower()))

    # Show only paths under base, in a tree-like list
    for p in created:
        rel = p.relative_to(base)
        indent = "  " * (len(rel.parts) - 1)
        print(f"{indent}- {rel.parts[-1]}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Sighnal backend folder structure.")
    parser.add_argument(
        "--base",
        default=r"D:\Hare Krishna_ai_blog",
        help=r'Base path where the project folder structure will be created (default: D:\Hare Krishna_ai_blog)',
    )
    args = parser.parse_args()

    base = Path(args.base)

    # Ensure base exists
    base.mkdir(parents=True, exist_ok=True)

    # Create required directories
    for d in DIRS:
        (base / d).mkdir(parents=True, exist_ok=True)

    print("✅ Folder structure created (or already existed).")
    _print_tree(base)


if __name__ == "__main__":
    main()
