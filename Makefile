PYTHON ?= python
CONFIG ?= configs/default.yaml
export PYTHONPATH := src

.PHONY: install test data spectrum pod report all

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) -m burgers.cli data --config $(CONFIG)

spectrum:
	$(PYTHON) -m burgers.cli spectrum --config $(CONFIG)

pod:
	$(PYTHON) -m burgers.cli pod --config $(CONFIG)

report:
	$(PYTHON) -m burgers.cli report --config $(CONFIG) --readme README.md

all:
	$(PYTHON) -m burgers.cli all --config $(CONFIG) --readme README.md
