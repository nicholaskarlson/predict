PYTHON ?= python3
.PHONY: verify r-audit boundary

boundary:
	$(PYTHON) scripts/verify_companion.py --boundary-only

verify: boundary
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/run_local_stages.py --stage verify

r-audit: boundary
	$(PYTHON) scripts/run_local_stages.py --stage base-r
