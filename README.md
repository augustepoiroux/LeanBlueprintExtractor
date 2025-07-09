# (WIP) LeanBlueprintExtractor

This repository features the code for the programmatic manipulation and data extraction of Lean 4 blueprint projects.

## Project setup

### Development inside Docker (recommended)

In VS Code, run the **Dev Containers: Open Folder in Container...** command from the Command Palette (F1). The `.devcontainer` folder contains the necessary configuration and will take care of setting up the environment.

### Local installation

Requirements:

- Python >= 3.10
- [Lean 4](https://leanprover-community.github.io/get_started.html)

Install Python project:

    pip install -e .

## Extracting a local project

To extract a local Lean project, you can use the `lean-blueprint-extract-local` command. This command will extract data from LaTeX and Lean files, and align them:

```bash
lean-blueprint-extract-local --project-dir /path/to/lean/project --nb-process 4
```

You can find the output in the `.trace_cache` directory within your project directory. The `blueprint_to_lean.jsonl` file contains the extracted and aligned data.

## MCP server

This project comes with an MCP server implementation. This MCP server implementation is in particular backing the [LeanBlueprintCopilot](https://github.com/augustepoiroux/LeanBlueprintCopilot) VS Code extension.
Install this project with the `mcp` optional dependency to enable the MCP server:

```bash
pip install -e .[mcp]
```

Set the directory of the Lean project you want to work with in the environment variable `LEAN_BLUEPRINT_PROJECT_DIR`. For example, if your Lean project is located at `/path/to/lean/project`, you can set the environment variable as follows:

```bash
export LEAN_BLUEPRINT_PROJECT_DIR="/path/to/lean/project"
```

Start the MCP server with:

```bash
lean-blueprint-mcp
```

The MCP server contains incremental state reuse (see this [PR](https://github.com/leanprover-community/repl/pull/110)), mirroring and extending the human VS Code experience for LLM agents. When editing a file, previous executions of the file will be reused, hence avoiding a complete re-execution of the file. Here, this feature takes into account the whole history of edited files, making backtracking and sampling more efficient, and cross-file execution sharing possible.
