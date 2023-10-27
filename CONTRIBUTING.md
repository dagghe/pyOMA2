# Contribution Guide

This project runs on python>=3.7,<3.11

## Setup

Create and activate a virtual environment

```shell
python3 -m venv .venv

# Unix
source .venv/bin/activate

# Windows cmd
.\.venv\Scripts\Activate.bat
# Windows Powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies

```shell
pip install -r requirements.txt
```

Install pre-commit

```shell
pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Conventions

### Commits

Use conventional commits guidelines https://www.conventionalcommits.org/en/v1.0.0/
