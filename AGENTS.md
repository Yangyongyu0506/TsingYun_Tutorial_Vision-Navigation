# Agent Notes

- Primary entrypoint: `python navigation/main.py --level 1 --record output.mp4` (run from repo root). Level 2 enables dynamic obstacles + lidar.
- Recommended env uses `uv`: `uv venv -p 3.11`, `source .venv/bin/activate`, `uv pip install -r requirements.txt`, then `uv run python navigation/main.py ...`.
- Assignment scope is limited to the four TODO functions in `navigation/nav/` (costmap, planner, controller). Avoid editing other files unless explicitly required; update `requirements.txt` if you add deps.
- Default settings live in `navigation/configs/cfg.yaml`; docs require restoring defaults before submission.
- Task details, controls, and algorithm guidance are in `docs/assignment.md`; prefer it over README when implementing nav logic.
