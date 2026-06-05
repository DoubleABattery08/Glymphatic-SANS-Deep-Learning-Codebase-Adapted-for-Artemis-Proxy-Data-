PYTHON ?= python

.PHONY: all nhanes dictionary cohorts validation control figures lint

all:
	$(PYTHON) scripts/run_all.py

nhanes:
	$(PYTHON) scripts/download_nhanes.py

dictionary:
	$(PYTHON) scripts/build_data_dictionary.py

cohorts:
	$(PYTHON) scripts/build_cohorts.py

validation:
	$(PYTHON) scripts/run_validation.py

control:
	$(PYTHON) scripts/integrate_control.py

figures:
	$(PYTHON) scripts/make_figures.py

lint:
	black --check src scripts
	ruff check src scripts
