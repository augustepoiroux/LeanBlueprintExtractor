import logging
import os
import re
from typing import Any

from lean_interact.utils import (
    compress_newlines,
    extract_last_theorem,
    lean_comments_ranges,
)
from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger("lean_blueprint_extractor")
logger.setLevel("INFO")
handler = RichHandler(rich_tracebacks=True)
handler.setLevel("NOTSET")
handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
logger.handlers = []
logger.addHandler(handler)

console = Console()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def extract_lean_codes(content: str) -> list[str]:
    """Extract Lean code snippets from a string, likely from a LLM response."""
    lean_starts = ["```lean", "```lean4", "```lean-4"]

    # If no lean code blocks found, return content as is
    if not any(start in content for start in lean_starts):
        return [content.strip()]

    lean_codes = []
    while any(start in content for start in lean_starts):
        # Find the earliest occurrence of any lean start marker
        start_indices = [(content.index(start), start) for start in lean_starts if start in content]
        idx_start, start_marker = min(start_indices, key=lambda x: (x[0], -len(x[1])))

        content = content[idx_start + len(start_marker) :]
        try:
            idx_end = content.index("```")
        except ValueError:
            idx_end = len(content)
        lean_codes.append(content[:idx_end].strip())
        content = content[idx_end + 3 :]
    return lean_codes


def extract_last_theorem_name(lean_code: str) -> str:
    """Extract the last theorem name from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem."""
    idx_last_theorem = extract_last_theorem(lean_code)
    theorem_decl_keywords = ["theorem", "lemma"]
    for keyword in theorem_decl_keywords:
        if lean_code[idx_last_theorem:].startswith(keyword):
            idx_theorem_name = idx_last_theorem + len(keyword)
            return lean_code[idx_theorem_name:].split()[0]

    raise ValueError(f"Theorem name extraction failed for the following Lean code:\n{lean_code}")


def remove_ranges(content: str, ranges: list[tuple[int, int]]) -> str:
    ranges = sorted(ranges, key=lambda x: x[0])

    # merge potential overlapping ranges
    merged_ranges = []
    for start, end in ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    new_content = ""
    prev_start = 0
    for start, end in merged_ranges:
        if start >= len(content):
            break
        if end >= len(content):
            end = len(content)
        new_content += content[prev_start:start]
        prev_start = end
    new_content += content[prev_start:]

    return new_content


def replace_ranges(content: str, ranges: list[tuple[int, int, str]]) -> str:
    ranges = sorted(ranges, key=lambda x: x[0])

    # check potential overlapping ranges
    for i in range(len(ranges) - 1):
        if ranges[i][1] > ranges[i + 1][0]:
            raise ValueError("Overlapping ranges")

    new_content = ""
    prev_start = 0
    for start, end, replacement in ranges:
        if start >= len(content):
            break
        if end >= len(content):
            end = len(content)
        new_content += content[prev_start:start] + replacement
        prev_start = end
    new_content += content[prev_start:]

    return new_content


def trim_comments_end(lean_code: str) -> str:
    """Trim the end of a Lean code snippet by removing everything that is commented or whitespace.
    We keep newlines at the end of the code snippet."""
    comment_ranges = lean_comments_ranges(lean_code)
    idx = len(lean_code)

    while idx > 0:
        idx -= 1
        if lean_code[idx].isspace():
            continue
        # Check if idx is inside a comment range
        in_comment = False
        for start, end in comment_ranges:
            if start <= idx < end:
                in_comment = True
                idx = start
                break
        if not in_comment:
            # Found non-comment, non-whitespace character
            idx += 1
            while idx < len(lean_code) and lean_code[idx] == "\n":
                idx += 1
            break
    return lean_code[:idx]


class LeanFilesProcessor:
    def __init__(self, lean_declarations: list[dict[str, Any]]):
        self.lean_declarations = lean_declarations

    def compress(self, file_content: str, remove_top_comment: bool = True) -> str:
        """Compress extra lines / whitespaces and remove top comments at the beginning of the file as \
        they often only contain information about the author and file license."""
        # remove top comments
        if remove_top_comment:
            comment_ranges = lean_comments_ranges(file_content)
            if len(comment_ranges) > 0 and comment_ranges[0][0] == 0:
                file_content = remove_ranges(file_content, [comment_ranges[0]])

        # remove all `#align` lines
        file_content = re.sub(r"^\s*#align.*$", "", file_content, flags=re.MULTILINE)

        # remove comments using /-%% format
        comment_ranges = lean_comments_ranges(
            file_content, multiline_comment_suffix="%", remove_single_line_comments=False
        )

        file_content = remove_ranges(file_content, comment_ranges)

        return compress_newlines(file_content)

    def remove_theorems(
        self,
        lean_file_path: str,
        lean_file_content: str,
        whitelist: set[str] | None = None,
        compress_newlines_comments: bool = True,
        remove_top_comment: bool = True,
    ) -> str:
        """Remove all theorems from a Lean file.
        Caution: this will very likely produce invalid Lean code"""
        if whitelist is None:
            whitelist = set()

        # collect all theorem declarations
        theorem_decls_ranges = []
        for decl in self.lean_declarations:
            if decl["file"] == lean_file_path and "theorem_info" in decl and decl["full_name"] not in whitelist:
                # before removing the theorem, we have to check if the previous non-empty line is an "open ... in"
                start_idx = decl["start_idx"]
                lines_before = lean_file_content[:start_idx].split("\n")
                i = len(lines_before) - 1
                while i >= 0 and not lines_before[i].strip():
                    i -= 1
                if i >= 0 and lines_before[i].rstrip().endswith(" in"):
                    nb_char = sum(len(line) + 1 for line in lines_before[i + 1 :])
                    nb_char += len(lines_before[i]) - lines_before[i].rfind(" in")
                    start_idx = start_idx - nb_char

                theorem_decls_ranges.append((start_idx, decl["end_idx"]))

        new_content = remove_ranges(lean_file_content, theorem_decls_ranges)

        if compress_newlines_comments:
            new_content = self.compress(new_content, remove_top_comment=remove_top_comment)

        return new_content

    def remove_proofs(
        self,
        lean_file_path: str,
        lean_file_content: str,
        whitelist: set[str] | None = None,
        compress_newlines: bool = True,
        remove_top_comment: bool = True,
    ) -> str:
        """Remove all proofs from a Lean file.
        Caution: this will very likely produce invalid Lean code"""
        if whitelist is None:
            whitelist = set()

        # collect all theorem declarations
        proof_ranges = []
        for decl in self.lean_declarations:
            if decl["file"] == lean_file_path and "theorem_info" in decl and decl["full_name"] not in whitelist:
                proof_ranges.append((decl["theorem_info"]["proof_start_idx"], decl["end_idx"], " := sorry"))

        new_content = replace_ranges(lean_file_content, proof_ranges)

        if compress_newlines:
            new_content = self.compress(new_content, remove_top_comment=remove_top_comment)

        return new_content
