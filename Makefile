PYTHON ?= python
CONFIG ?= configs/default.yaml
export PYTHONPATH := src

.PHONY: install test data spectrum pod report all phase2-smoke phase2-time phase2-gate phase2-tierA phase2-tierB phase2-tierC phase2-aggregate phase2-report

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

# ---- Phase 2 (needs: pip install -e .[phase2]) ----
JOBS ?= 4
ARGS ?=
P2 = $(PYTHON) -m burgers.pinn.cli2
P2DIR = results/phase2
# print the last 20 lines of a log (portable: works under sh and cmd)
TAIL = $(PYTHON) -c "import sys, collections; print(''.join(collections.deque(open(sys.argv[1]), 20)), end='')"
MKDIR = $(PYTHON) -c "import os; os.makedirs('$(P2DIR)', exist_ok=True)"

phase2-smoke:
	$(P2) smoke

phase2-time:
	$(P2) time --jobs $(JOBS)

phase2-gate:
	$(MKDIR)
	-$(P2) gate --jobs $(JOBS) $(ARGS) > $(P2DIR)/run_gate.log 2>&1
	$(TAIL) $(P2DIR)/run_gate.log

phase2-tierA:
	$(MKDIR)
	-$(P2) run --tier A --jobs $(JOBS) $(ARGS) > $(P2DIR)/run_A.log 2>&1
	$(TAIL) $(P2DIR)/run_A.log

phase2-tierB:
	$(MKDIR)
	-$(P2) run --tier B --jobs $(JOBS) $(ARGS) > $(P2DIR)/run_B.log 2>&1
	$(TAIL) $(P2DIR)/run_B.log

phase2-tierC:
	$(MKDIR)
	-$(P2) run --tier C --jobs $(JOBS) $(ARGS) > $(P2DIR)/run_C.log 2>&1
	$(TAIL) $(P2DIR)/run_C.log

phase2-aggregate:
	$(P2) aggregate

phase2-report:
	$(P2) report --readme README.md
