# Contributing

Thank you for your interest in this project. As a hackathon submission it is
primarily a portfolio piece, but pull requests are welcome for bug fixes,
documentation improvements, and new fraud detection features.

## Setup

```bash
git clone https://github.com/aliivaezii/multi-agent-fraud-detection.git
cd multi-agent-fraud-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your OpenRouter and Langfuse keys
```

## Running tests

```bash
pytest tests/ -v
```

The test suite mocks all LLM calls and requires no API keys. Two smoke tests
(`TestPipelineSmoke`) are automatically skipped when competition data ZIPs
are absent from `train-validation/`.

## Branching

| Branch pattern | Purpose |
|---|---|
| `main` | Stable, always green |
| `feat/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation only |

Please open a branch and PR rather than committing directly to `main`.

## Code style

- Readability over cleverness — no one-liner tricks
- Comments answer *why*, not *what*
- Max line length: 88 characters (Black-compatible)
- All new functions need at least a one-line docstring

## Reporting issues

Use the GitHub issue templates. Do **not** include API keys, competition data,
or personal credentials in issues or PRs.
