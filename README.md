# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: RAPIDS Analytics Frameworks Toolset</div>

RAFT is a repository containining shared utilities, mathematical operations and common functions for the analytics components of RAPIDS. Both the C++ and Python components can be included in consuming libraries.

Refer to the [Build and Development Guide](Build.md) for details on RAFT's design, building, testing and development guidelines.

## Folder Structure and Contents

The folder structure mirrors the main RAPIDS repos (cuDF, cuML, cuGraph...), with the following folders:

- `cpp`: Source code for all C++ code. The code is header only, therefore it is in the `include` folder (with no `src`).
- `python`: Source code for all Python source code.
- `ci`: Scripts for running CI in PRs


