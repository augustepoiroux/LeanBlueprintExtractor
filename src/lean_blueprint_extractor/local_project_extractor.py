import argparse
import os
import sys
from pathlib import Path

import jsonlines
from lean_interact import LocalProject

from lean_blueprint_extractor.blueprint_extractor import (
    extract_blueprint_info,
)
from lean_blueprint_extractor.blueprint_lean_linking import (
    merge_blueprint_lean_dep_graphs,
)
from lean_blueprint_extractor.lean_extractor import trace_repo


def parse_local_project(project_dir: str, nb_process: int, repl_cache_dir: Path | None = None) -> bool:
    """
    Parse a local Lean project directory to extract blueprint information.

    Args:
        project_dir (str): Path to the local Lean project directory.
        nb_process (int): Number of processes to use for tracing.
        repl_cache_dir (Path | None): Optional path for REPL cache directory.

    Returns:
        bool: True if the parsing was successful, False otherwise.
    """
    try:
        project = LocalProject(project_dir)
        project_path, declarations = trace_repo(project, nb_process, repl_cache_dir=repl_cache_dir)
        trace_dir = project_path / ".cache" / "blueprint_trace"

        blueprint_src_path = project_path / "blueprint"
        blueprint_graph = extract_blueprint_info(blueprint_src_path)

        with jsonlines.open(trace_dir / "blueprint.jsonl", "w") as writer:
            writer.write_all(blueprint_graph)

        dep_graph_info = merge_blueprint_lean_dep_graphs(blueprint_graph, declarations)

        with jsonlines.open(trace_dir / "blueprint_to_lean.jsonl", "w") as writer:
            writer.write_all(dep_graph_info)
        return True

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point for the local project extractor command."""
    parser = argparse.ArgumentParser(description="Extract blueprint info from a local Lean project")
    parser.add_argument("--project-dir", type=str, default=".", help="Path to the project directory")
    default_nb_process = max((os.cpu_count() or 1) // 2, 1)
    parser.add_argument(
        "--nb-process", type=int, default=default_nb_process, help="Number of processes to use for tracing"
    )
    args = parser.parse_args()

    repl_cache_dir = Path(args.project_dir) / ".cache" / "lean_interact"
    parse_local_project(args.project_dir, args.nb_process, repl_cache_dir=repl_cache_dir)


if __name__ == "__main__":
    main()
