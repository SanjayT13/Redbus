# Repository AI instructions for Copilot / AI coding agents

Purpose: Give concise, repo-specific guidance so AI agents can make safe, runnable, and reviewable changes.

Quick run & test commands
- `python -m pip install -U pip setuptools wheel`
- `python -m pip install -r requirements.txt`
- `python test_environment.py`
- `make requirements` / `make data` / `make lint` (from repo root)
- `pytest -q` (when tests exist)

Primary places to inspect
- `src/data/make_dataset.py` — dataset ingestion and pipeline entrypoints
- `src/features/build_features.py` — feature engineering
- `src/models/train_model.py`, `src/models/predict_model.py` — training & inference
- `params.yaml` — canonical hyperparameters and pipeline options
- `Makefile`, `test_environment.py` — environment setup and sanity checks
- `dvc.yaml` and `data/` — data tracked by DVC (do not commit raw data)

Project-specific conventions & gotchas
- Data is DVC-managed: do not add large raw data files directly. Prefer DVC operations and open PRs describing data changes.
- Exact path names matter: `data/encodding_files/` contains a spelling typo — reference it exactly.
- Avoid creating folders or paths with spaces (repo already contains `logs/pred model/`).
- The Makefile contains template remnants: `PROJECT_NAME := nyc_taxi` and `PYTHON_INTERPRETER = python3`. Confirm renames before editing env scripts.
- External integrations (S3/AWS CLI) are referenced in Makefile targets — mock/stub these in tests.

Patch & commit style
- Keep patches small and focused. Include a test or smoke check for behavioral changes.
- Branch name for AI-generated work: `autogen/<short-description>`.
- PR body: 1–2 line summary and mention any `params.yaml` or `dvc.yaml` changes.

Testing & CI expectations
- Run `python test_environment.py` before applying edits.
- Run `make lint` (flake8) for style checks.
- Tests must not require downloading large datasets or credentials; use fixtures/mocks.

When to escalate to humans
- Changes that modify `dvc.yaml`, data schema, production credentials, or model input/output contracts.
- Adding/removing heavy files (>10MB) or touching production S3/DVC pipelines.

Example prompt (copyable)
"Edit `src/features/build_features.py` to add `def build_features(df)` returning a `DataFrame` with columns `['x','y']`. Add `tests/test_build_features.py` using a small fixture and assert output columns. Run `python test_environment.py` locally before applying the patch. Do not modify DVC or add data files."

Safety
- Never include secrets or credentials in patches. Use placeholders and environment variables for examples.
- If a change requires real credentials or large data, abort and request human approval.

Keep this file short and specific — it should allow an AI agent to make correct, small changes without repeated clarification.
