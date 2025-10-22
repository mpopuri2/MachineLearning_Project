.PHONY: venv install train eval live lint test

venv:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip

install:
	. .venv/bin/activate && pip install -r requirements.txt

train:
	. .venv/bin/activate && python -m src.train

eval:
	. .venv/bin/activate && python -m src.evaluate

live:
	. .venv/bin/activate && python -m src.live_trade

lint:
	. .venv/bin/activate && python -m pip install ruff && ruff check src

test:
	. .venv/bin/activate && pytest -q