# (WIP) LeanBlueprintExtractor

This repository features the code for the programmatic manipulation and data extraction of Lean 4 blueprint projects.
Compatible with all Lean versions between 4.8.0-rc1 and v4.22.0-rc3.

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

You can find the output in the `.cache/blueprint_trace` directory within your project directory. The `blueprint_to_lean.jsonl` file contains the extracted and aligned data.

## MCP server

This project comes with an MCP server implementation. This MCP server implementation is in particular backing the [LeanBlueprintCopilot](https://github.com/augustepoiroux/LeanBlueprintCopilot) VS Code extension.

The MCP server contains incremental state reuse (see this [PR](https://github.com/leanprover-community/repl/pull/110)), mirroring and extending the human VS Code experience for LLM agents. When editing a file, previous executions of the file will be reused, hence avoiding a complete re-execution of the file. Here, this feature takes into account the whole history of edited files, making backtracking and sampling more efficient, and cross-file execution sharing possible.

### Gemini-CLI

Follow general gemini-cli instructions [here](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md), and add the following configuration to your `.gemini/settings.json` file:

```json
"lean-blueprint-mcp": {
    "command": "bash",
    "args": [
        "-c",
        "uvx --from git+https://github.com/augustepoiroux/LeanBlueprintExtractor[mcp] lean-blueprint-mcp"
    ],
    "env": {
        "LEAN_BLUEPRINT_PROJECT_DIR": "/home/poiroux/Documents/EPFL/PhD/lean_blueprint_ai_vscode/FLT"
    }
}
```

### VS Code

Follow the instructions on the [VS Code website](https://code.visualstudio.com/docs/copilot/chat/mcp-servers) and install [uv](https://docs.astral.sh/uv/getting-started/installation/). Here is the configuration you can use:

```json
"lean-blueprint-mcp": {
    "type": "stdio",
    "command": "bash",
    "args": ["-c", "uvx --from git+https://github.com/augustepoiroux/LeanBlueprintExtractor[mcp] lean-blueprint-mcp"],
    "env": {
        "LEAN_BLUEPRINT_PROJECT_DIR": "path/to/lean/project"
    }
}
```

### Manually launching the MCP server with [`uvx`](https://docs.astral.sh/uv/guides/tools/)

Run the following command to start the MCP server with the Lean project you want to work with. Make sure to replace `path/to/lean/project` with the actual path to your Lean project:

```bash
export LEAN_BLUEPRINT_PROJECT_DIR="path/to/lean/project"
uvx --from git+https://github.com/augustepoiroux/LeanBlueprintExtractor[mcp] lean-blueprint-mcp
```

You can pass arguments to the MCP server by appending them after `lean-blueprint-mcp`: `... lean-blueprint-mcp --port 5000`.

### Manually launching the MCP server with local installation

Alternatively, install this project with the `mcp` optional dependency to enable the MCP server:

```bash
pip install -e .[mcp]
export LEAN_BLUEPRINT_PROJECT_DIR="/path/to/lean/project"
lean-blueprint-mcp
```
