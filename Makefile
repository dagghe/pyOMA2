# Makefile

test:
    pdm run pytest

test-coverage:
	pdm run pytest --cov=pyoma2 --cov-report=html

env38:
	pdm venv create -n venv38
	pdm install --venv venv38
	# then with "pdm use" select the created venv

tox:
	pdm run tox

export:
	pdm export --without-hashes -L pdm.lock -v >> requirements.txt


.PHONY: test test-coverage env38 tox export
