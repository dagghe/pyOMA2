# Makefile

test:
	uv run pytest

test-coverage:
	uv run pytest --cov=pyoma2 --cov-report=html

tox:
	uv run tox

export:
	uv export --no-hashes -o requirements.txt

.PHONY: test test-coverage tox export
