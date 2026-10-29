# Original-project comparison audit

Compared this codebase against the user-provided `RoboPilot-Sim2Real.zip`.

## File-level result

All original non-secret project files are present in this archive. The comparison found no missing original project files after restoring the missing files/directories.

Intentionally not copied from the uploaded archive:
- `.git/`: repository metadata and history, not application source files.
- `.idea/`: IDE-specific settings.
- `.env` files: local credentials/secrets are not copied into distributable archives.
- Python bytecode and cache directories.

The following original paths are preserved, including the previously missed paths:
- `data/adapter/raw/` and `data/adapter/standardized/`
- `runs/`
- `web/`
- root `main.py`
- `test_api.ipynb`
- original robot assets and simulation tests
- evaluation helpers, telemetry and simulation helpers
- training deployment, replay-buffer, rollout, reward and synthetic-data files

## Test coverage restored

The rebuilt version had reduced `tests/test_data_adapter.py` to a single test. The original four test cases have now been restored, covering:
- generic transition records and persistence
- episode-oriented input payloads
- rejecting records without actions
- coordinate transforms and embodiment/joint mapping

Validation after restoration:
- Python syntax compilation passed.
- `pytest -q`: **19 passed**.

## Important distinction: present does not mean unchanged

The rebuild intentionally changes content in a number of files rather than preserving the original implementation byte-for-byte. The main rewritten areas include:
- runtime task orchestration, planner validation and control
- MuJoCo runtime scene construction
- data-adapter API integration
- root launcher, web UI wiring and API smoke-test notebook
- the training RL environment and world-model implementation
- root and training documentation

The original `training/deploy/` source files are present, but the deployment README was simplified during the rebuild. Their container/service flow should be reviewed and tested separately before relying on it; file presence alone does not prove the old deployment workflow still works end-to-end.

The runtime and training codebases remain intentionally separate: `training/` must not import `app/`, call runtime APIs, or use the runtime LLM planner for training data, labels, rewards or policy learning.
