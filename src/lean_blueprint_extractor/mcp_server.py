import json
import os
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from lean_interact import AutoLeanServer, FileCommand, LeanREPLConfig, LocalProject
from lean_interact.interface import CommandResponse, LeanError
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import base

from lean_blueprint_extractor.local_project_extractor import parse_local_project


@dataclass
class AppContext:
    repl_config: LeanREPLConfig
    repl_server: AutoLeanServer
    project_dir: Path
    nb_process_parsing: int


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    project_dir = Path(os.environ.get("LEAN_BLUEPRINT_PROJECT_DIR", "")).resolve()
    nb_process_parsing = int(os.environ.get("LEAN_BLUEPRINT_NB_PROCESS", 0))
    if not project_dir:
        raise ValueError("Environment variable LEAN_BLUEPRINT_PROJECT_DIR is not set.")

    try:
        repl_config = LeanREPLConfig(
            project=LocalProject(directory=project_dir),
            repl_cache_dir=project_dir / ".cache" / "lean_interact",
        )
        repl_server = AutoLeanServer(config=repl_config)
        yield AppContext(
            repl_config=repl_config,
            repl_server=repl_server,
            project_dir=project_dir,
            nb_process_parsing=nb_process_parsing,
        )
    finally:
        pass


mcp = FastMCP(
    "LeanBlueprintCopilot",
    description="Lean Blueprint Copilot",
    dependencies=["lean-interact"],
    lifespan=app_lifespan,
    env_vars={
        "LEAN_BLUEPRINT_PROJECT_DIR": {
            "description": "Path to the Lean project directory",
            "required": True,
        },
        "LEAN_BLUEPRINT_NB_PROCESS": {
            "description": "Number of processes to use for parsing Lean files",
            "default": str(max((os.cpu_count() or 1) // 2, 1)),
        },
    },
)


@mcp.tool()
async def rebuild_project(ctx: Context) -> str:
    """Rebuild the Lean project. Use sparingly.
    May be necessary when multiple interdependent files have been added/modified."""
    await ctx.info("Rebuilding the Lean project...")
    await ctx.report_progress(0, 1)
    try:
        subprocess.run(
            ["lake", "build"],
            cwd=ctx.request_context.lifespan_context.project_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await ctx.info("Build completed successfully.")
        return "Build completed successfully."
    except subprocess.CalledProcessError as e:
        await ctx.error(f"Build failed: {e.stderr.decode().strip()}")
        return f"Build failed: {e.stderr.decode().strip()}"


@mcp.tool()
async def restart_lean(ctx: Context) -> None:
    """Restart the Lean server. Only useful if you are using the `run_file` tool."""
    await ctx.request_context.lifespan_context.repl_server.restart()
    await ctx.info("Lean server restarted successfully.")


@mcp.tool()
async def reparse_project(ctx: Context) -> None:
    """Re-parse the project. Use sparingly.
    Useful if you have made several changes to the Lean files and/or LaTeX files."""
    await ctx.info("Re-parsing the project...")
    await ctx.report_progress(0, 1)
    try:
        if parse_local_project(
            ctx.request_context.lifespan_context.project_dir,
            ctx.request_context.lifespan_context.nb_process_parsing,
            repl_cache_dir=ctx.request_context.lifespan_context.repl_config._cache_repl_dir,
        ):
            await ctx.info("Re-parsing completed successfully.")
        else:
            await ctx.error("Re-parsing failed. Please check the logs for more details.")
    except Exception as e:
        await ctx.error(f"Re-parsing failed: {e}")


@mcp.tool()
async def run_file(ctx: Context, file: str) -> CommandResponse | LeanError:
    """Run a specific Lean file and return the result.
    To be used only if you don't have access to VS Code or Lean extension output."""
    await ctx.report_progress(0, 1)
    try:
        lean_file = (ctx.request_context.lifespan_context.project_dir / file).resolve()
        repl_server: AutoLeanServer = ctx.request_context.lifespan_context.repl_server
        file_res = await repl_server.async_run(FileCommand(path=str(lean_file)))
        if isinstance(file_res, LeanError):
            await ctx.error(f"Provided file `{file}` could not be loaded: {file_res}")
        return file_res
    except Exception as e:
        await ctx.error(f"Type checking failed with unexpected error: {e}")
        raise e


@mcp.prompt()
def incorporate_latex_to_blueprint(raw_latex: str) -> list[base.Message]:
    """Incorporate raw Latex into Lean blueprint format using a LLM"""
    return [
        base.UserMessage("""You are an expert in Lean blueprints. Given the following raw LaTeX, structure it using the Lean blueprint format and incorporate the result in the blueprint. The Lean Blueprint format is characterized in particular by the following macros:
* `\\lean` that lists the Lean declaration names corresponding to the surrounding
    definition or statement (including namespaces).
* `\\leanok` which claims the surrounding environment is fully formalized. Here
    an environment could be either a definition/statement or a proof. You won't
    use this macro here as the content is not formalized yet.
* `\\uses` that lists LaTeX labels that are used in the surrounding environment.
    This information is used to create the dependency graph. Here
    an environment could be either a definition/statement or a proof, depending on
    whether the referenced labels are necessary to state the definition/theorem
    or only in the proof.

The example below show those essential macros in action, assuming the existence of
LaTeX labels `def:immersion`, `thm:open_ample` and `lem:open_ample_immersion` and
assuming the existence of a Lean declaration `sphere_eversion`.

```latex
\\begin{theorem}[Smale 1958]
    \\label{thm:sphere_eversion}
    \\lean{sphere_eversion}
    \\uses{def:immersion}
    There is a homotopy of immersions of $𝕊^2$ into $ℝ^3$ from the inclusion map to
    the antipodal map $a : q ↦ -q$.
\\end{theorem}

\\begin{proof}
    \\uses{thm:open_ample, lem:open_ample_immersion}
    This obviously follows from what we did so far.
\\end{proof}
```

Note that the proof above is abbreviated in this documentation.
Be nice to you and your collaborators and include more details in your blueprint proofs!"""),
        base.UserMessage(
            f"Here is the raw latex content you should incorporate into the blueprint of the project:\n```latex\n{raw_latex}\n```"
        ),
        base.UserMessage(
            """Transform the provided raw LaTeX into structured Lean blueprint LaTeX, and update the blueprint. The main content of your blueprint should live in `blueprint/src/content.tex` (or in files imported in `content.tex` if you want to split your content)."""
        ),
    ]


@mcp.tool()
async def get_blueprint_node_context(ctx: Context, node_label: str) -> str:
    """Get detailed context for a blueprint node including dependencies and formalization status"""
    await ctx.info(f"Getting context for blueprint node: {node_label}")
    await ctx.report_progress(0, 1)

    try:
        project_dir: Path = ctx.request_context.lifespan_context.project_dir
        blueprint_data_file = project_dir / ".cache" / "blueprint_trace" / "blueprint_to_lean.jsonl"

        if not blueprint_data_file.exists():
            return "Blueprint data not found. Please run 'Parse Blueprint Project' first."

        # Load and parse blueprint data
        nodes = []
        with open(blueprint_data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        nodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Find the target node
        target_node = None
        for node in nodes:
            if node.get("label") == node_label:
                target_node = node
                break

        if not target_node:
            return f"Node with label '{node_label}' not found in blueprint data."

        # Build context information
        context_info = {
            "node": target_node,
            "formalization_status": {
                "is_formalized": bool(target_node.get("leanok") or target_node.get("fully_proved")),
                "has_lean_declarations": bool(target_node.get("lean_declarations")),
                "can_state": target_node.get("can_state", False),
                "can_prove": target_node.get("can_prove", False),
                "proved": target_node.get("proved", False),
            },
            "dependencies": [],
            "dependents": [],
        }

        # Find dependencies (nodes this one uses)
        if target_node.get("proof") and target_node["proof"].get("uses"):
            for dep_label in target_node["proof"]["uses"]:
                for node in nodes:
                    if node.get("label") == dep_label:
                        context_info["dependencies"].append(
                            {
                                "label": dep_label,
                                "type": node.get("stmt_type"),
                                "is_formalized": bool(node.get("leanok") or node.get("fully_proved")),
                                "text": node.get("processed_text", "")[:200] + "..."
                                if len(node.get("processed_text", "")) > 200
                                else node.get("processed_text", ""),
                            }
                        )
                        break

        # Find dependents (nodes that use this one)
        for node in nodes:
            if node.get("proof") and node["proof"].get("uses"):
                if node_label in node["proof"]["uses"]:
                    context_info["dependents"].append(
                        {
                            "label": node.get("label"),
                            "type": node.get("stmt_type"),
                            "is_formalized": bool(node.get("leanok") or node.get("fully_proved")),
                            "text": node.get("processed_text", "")[:200] + "..."
                            if len(node.get("processed_text", "")) > 200
                            else node.get("processed_text", ""),
                        }
                    )

        return json.dumps(context_info, indent=2)

    except Exception as e:
        await ctx.error(f"Failed to get blueprint node context: {e}")
        return f"Error getting context: {e}"


@mcp.tool()
async def get_node_dependencies(ctx: Context, node_label: str) -> str:
    """Get the dependency tree for a specific blueprint node"""
    await ctx.info(f"Getting dependencies for node: {node_label}")
    await ctx.report_progress(0, 1)

    try:
        project_dir: Path = ctx.request_context.lifespan_context.project_dir
        blueprint_data_file = project_dir / ".cache" / "blueprint_trace" / "blueprint_to_lean.jsonl"

        if not blueprint_data_file.exists():
            return "Blueprint data not found. Please run 'Parse Blueprint Project' first."

        # Load blueprint data
        nodes = {}
        with open(blueprint_data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        node = json.loads(line)
                        if node.get("label"):
                            nodes[node["label"]] = node
                    except json.JSONDecodeError:
                        continue

        def get_deps_recursive(label, visited=None):
            if visited is None:
                visited = set()
            if label in visited:
                return {"label": label, "circular": True}
            if label not in nodes:
                return {"label": label, "not_found": True}

            visited.add(label)
            node = nodes[label]
            deps = []

            if node.get("proof") and node["proof"].get("uses"):
                for dep_label in node["proof"]["uses"]:
                    deps.append(get_deps_recursive(dep_label, visited.copy()))

            visited.remove(label)
            return {
                "label": label,
                "type": node.get("stmt_type"),
                "is_formalized": bool(node.get("leanok") or node.get("fully_proved")),
                "text": node.get("processed_text", "")[:100] + "..."
                if len(node.get("processed_text", "")) > 100
                else node.get("processed_text", ""),
                "dependencies": deps,
            }

        dependency_tree = get_deps_recursive(node_label)
        return json.dumps(dependency_tree, indent=2)

    except Exception as e:
        await ctx.error(f"Failed to get node dependencies: {e}")
        return f"Error getting dependencies: {e}"


@mcp.tool()
async def get_related_lean_code(ctx: Context, node_label: str) -> str:
    """Get related Lean code for a blueprint node and its dependencies"""
    await ctx.info(f"Getting related Lean code for node: {node_label}")
    await ctx.report_progress(0, 1)

    try:
        project_dir: Path = ctx.request_context.lifespan_context.project_dir
        blueprint_data_file = project_dir / ".cache" / "blueprint_trace" / "blueprint_to_lean.jsonl"

        if not blueprint_data_file.exists():
            await ctx.error("Blueprint data not found. Please run 'Parse Blueprint Project' first.")
            return "Blueprint data not found. Please run 'Parse Blueprint Project' first."

        # Load blueprint data
        nodes = {}
        with open(blueprint_data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        node = json.loads(line)
                        if node.get("label"):
                            nodes[node["label"]] = node
                    except json.JSONDecodeError:
                        continue

        if node_label not in nodes:
            return f"Node '{node_label}' not found in blueprint data."

        target_node = nodes[node_label]
        related_code = {
            "target_node": {
                "label": node_label,
                "lean_names": target_node.get("lean_names", []),
                "lean_declarations": [],
            },
            "dependencies": [],
        }

        # Get Lean declarations for target node
        if target_node.get("lean_declarations"):
            for decl in target_node["lean_declarations"]:
                related_code["target_node"]["lean_declarations"].append(
                    {
                        "name": decl.get("full_name"),
                        "file": decl.get("real_file"),
                        "type": decl.get("pp_no_docstring", ""),
                        "range": decl.get("range"),
                    }
                )

        # Get Lean code for dependencies
        if target_node.get("proof") and target_node["proof"].get("uses"):
            for dep_label in target_node["proof"]["uses"]:
                if dep_label in nodes:
                    dep_node = nodes[dep_label]
                    dep_info = {
                        "label": dep_label,
                        "lean_names": dep_node.get("lean_names", []),
                        "lean_declarations": [],
                    }

                    if dep_node.get("lean_declarations"):
                        for decl in dep_node["lean_declarations"]:
                            dep_info["lean_declarations"].append(
                                {
                                    "name": decl.get("full_name"),
                                    "file": decl.get("real_file"),
                                    "type": decl.get("pp_no_docstring", ""),
                                    "range": decl.get("range"),
                                }
                            )

                    related_code["dependencies"].append(dep_info)

        return json.dumps(related_code, indent=2)

    except Exception as e:
        await ctx.error(f"Failed to get related Lean code: {e}")
        return f"Error getting related Lean code: {e}"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
