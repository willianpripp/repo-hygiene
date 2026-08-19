.DEFAULT_GOAL := help

.PHONY: help check self-test

help:
	@echo "Targets:"
	@echo "  check       Run the auditor against this repo (public-portfolio profile)"
	@echo "  self-test   Run the offline fixture assertions (no network, no gh)"
	@echo "  help        Show this message (default)"

check:
	python3 scripts/audit.py --here --profile public-portfolio

self-test:
	python3 scripts/audit.py --self-test
