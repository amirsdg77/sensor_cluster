# sensorcluster — common workflows. Run `make <target>` from a shell that has
# `make` available (Git Bash, WSL, mac/linux). Prefer `make demo` for a fresh
# end-to-end run; the rest are convenience aliases over `uv run sensorcluster …`.

UV   ?= uv
PORT ?= 8000

.PHONY: install lint test cov train serve demo

install:
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check . && $(UV) run ruff format --check .

test:
	$(UV) run pytest -q

cov:
	$(UV) run pytest --cov=src/sensorcluster --cov-report=term-missing --cov-fail-under=80

train:
	$(UV) run sensorcluster train --config configs/base.yaml

serve:
	$(UV) run sensorcluster serve --port $(PORT)

demo: install train serve
