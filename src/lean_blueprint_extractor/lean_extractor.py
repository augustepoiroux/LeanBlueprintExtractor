import glob
import os
import shutil
import traceback
from pathlib import Path
from time import perf_counter

import jsonlines
from joblib import Parallel, delayed
from lean_interact import AutoLeanServer, FileCommand, LeanREPLConfig
from lean_interact.config import BaseProject
from lean_interact.interface import DeclarationInfo, LeanError, Pos
from tqdm import tqdm

from lean_blueprint_extractor.utils import logger


def pos_to_index(lean_content: str, pos: tuple[int, int] | Pos) -> int:
    lines = lean_content.splitlines()
    if isinstance(pos, Pos):
        line, col = pos.line, pos.column
    else:
        line, col = pos
    return sum(len(line) + 1 for line in lines[: line - 1]) + col - 1


def index_to_pos(lean_content: str, idx: int) -> Pos:
    lines = lean_content.splitlines()
    line = 0
    col = idx
    while col >= len(lines[line]):
        col -= len(lines[line]) + 1
        line += 1
    return Pos(line=line + 1, column=col + 1)


def get_decl_pp_without_docstring(decl: DeclarationInfo) -> str:
    """Get the pretty-printed declaration without the docstring."""
    if decl.modifiers.doc_string:
        return decl.pp.replace(decl.modifiers.doc_string.content, "").strip()
    return decl.pp.strip()


def get_declaration_dependencies(
    lean_real_file: Path, lean_cached_file: Path, decl_traces: list[DeclarationInfo], lean_traced_content: str
) -> list[dict]:
    """Extract declaration dependencies from the traced Lean file."""
    results = []
    visited_full_name_debug = set()  # temporary as some declarations are extracted multiple times
    for decl_info in decl_traces:
        if decl_info.full_name in visited_full_name_debug:
            logger.warning(f"Declaration {decl_info.full_name} already processed, skipping.")
            continue
        visited_full_name_debug.add(decl_info.full_name)
        decl_dict = decl_info.model_dump(mode="json")
        res_dict = {
            "real_file": str(lean_real_file.resolve()),
            "cached_file": str(lean_cached_file.resolve()),
            "start_idx": pos_to_index(lean_traced_content, decl_info.range.start),
            "end_idx": pos_to_index(lean_traced_content, decl_info.range.finish),
            "pp_no_docstring": get_decl_pp_without_docstring(decl_info),
            **decl_dict,
        }
        results.append(res_dict)
    return results


def extract_declaration_dependencies(traced_project: Path) -> list[dict]:
    """Extract all Lean declaration identifiers from the traced repo and their dependencies."""
    declaration_dependencies = []
    for lean_cached_file in (traced_project.resolve() / ".cache" / "blueprint_trace").glob("**/*.lean"):
        trace_file = lean_cached_file.with_suffix(".declarations.jsonl")
        lean_real_file = traced_project / lean_cached_file.relative_to(traced_project / ".cache" / "blueprint_trace")
        if not lean_cached_file.exists():
            logger.error(f"{lean_cached_file} does not exist, skipping.")
            continue
        if not lean_real_file.exists():
            logger.warning(f"{lean_real_file} does not exist anymore.")

        lean_content = lean_cached_file.read_text(encoding="utf-8")

        decl_traces = []
        with jsonlines.open(trace_file, mode="r") as reader:
            for raw_declaration in reader:
                decl_info = DeclarationInfo(**raw_declaration)
                decl_traces.append(decl_info)

        declaration_dependencies.extend(
            get_declaration_dependencies(lean_real_file, lean_cached_file, decl_traces, lean_content)
        )
    return declaration_dependencies


def process_file(file: str, project_dir: Path, repl_config: LeanREPLConfig) -> tuple[str, bool, str | None, float]:
    # TODO: improve the caching strategy to take into account the dependency graph.
    # Modifications in imported files should trigger reprocessing of dependent files.

    start = perf_counter()
    file_path = Path(file)
    src_lean_file = project_dir / file_path
    dest_lean_file = project_dir / ".cache" / "blueprint_trace" / file_path.with_suffix(".lean")
    # If the file exists in .cache/blueprint_trace and content is the same, skip processing
    if dest_lean_file.exists():
        with open(src_lean_file, "rb") as f1, open(dest_lean_file, "rb") as f2:
            if f1.read() == f2.read():
                elapsed = 0.0
                return (str(file_path), True, None, elapsed)  # Skip processing

    server = AutoLeanServer(config=repl_config)
    try:
        response = server.run(FileCommand(path=str(file_path), declarations=True))  # type: ignore
        if isinstance(response, LeanError):
            raise ValueError(f"Error in file {file_path}: {response.message}")
    except Exception:
        tb = traceback.format_exc()
        elapsed = perf_counter() - start
        return (str(file_path), False, tb, elapsed)

    output_file = project_dir / ".cache" / "blueprint_trace" / file_path.with_suffix(".declarations.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("")

    declarations = getattr(response, "declarations", [])
    for decl in declarations:
        try:
            valid_decl = DeclarationInfo(**decl)
            with jsonlines.open(output_file, mode="a") as writer:
                writer.write(valid_decl.model_dump(by_alias=True, mode="json"))
        except Exception:
            tb = traceback.format_exc()
            elapsed = perf_counter() - start
            return (str(file_path), False, tb, elapsed)

    # Copy the Lean file to the .cache/blueprint_trace directory
    dest_lean_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_lean_file, dest_lean_file)

    elapsed = perf_counter() - start
    return (str(file_path), True, None, elapsed)


def trace_repo(
    project: BaseProject,
    nb_process: int,
    verbose: bool = True,
    repl_cache_dir: str | os.PathLike | None = None,
    project_cache_dir: str | os.PathLike | None = None,
) -> tuple[Path, list[dict]]:
    extra_attributes = {}
    if repl_cache_dir is not None:
        extra_attributes["repl_cache_dir"] = str(repl_cache_dir)
    if project_cache_dir is not None:
        extra_attributes["project_cache_dir"] = str(project_cache_dir)
    repl_config = LeanREPLConfig(
        project=project,
        verbose=verbose,
        **extra_attributes,
    )

    project_dir = Path(repl_config.working_dir)
    trace_dir = project_dir / ".cache" / "blueprint_trace"

    lean_files = glob.glob("**/*.lean", recursive=True, root_dir=project_dir)

    results = Parallel(n_jobs=nb_process, backend="multiprocessing")(
        delayed(process_file)(file, project_dir, repl_config) for file in tqdm(lean_files, desc="Processing files")
    )

    if verbose:
        # For summary: file -> (success, tb, elapsed)
        file_results = {}
        for file, result in zip(lean_files, results):
            if result is None:
                continue
            file_path, success, tb, elapsed = result
            file_results[file_path] = (success, tb, elapsed)

        # Log summary info
        logger.info("Processing Summary:")
        for file in lean_files:
            res = file_results.get(str(Path(file)), None)
            if res is None:
                logger.info(f"{file}: status=-, time=-")
            else:
                success, tb, elapsed = res
                if success:
                    logger.info(f"{file}: status=SUCCESS, time={elapsed:.2f}s")
                else:
                    logger.info(f"{file}: status=FAIL, time={elapsed:.2f}s")

        # Log errors if any
        errors = [(f, tb, t) for f, (s, tb, t) in file_results.items() if not s]
        if errors:
            logger.info("Files with errors:")
            for ef, tb, t in errors:
                logger.info(f"- {ef} (time: {t:.2f}s)\nTraceback:\n{tb}")
        else:
            logger.info("No errors encountered.")

    all_declarations = extract_declaration_dependencies(project_dir)
    with jsonlines.open(os.path.join(trace_dir, "lean_declarations.jsonl"), "w") as writer:
        writer.write_all(all_declarations)

    return project_dir, all_declarations
