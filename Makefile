.PHONY: setup test lint notebook demo clean

setup:
	python3 -m venv venv
	venv/bin/pip install -r requirements-dev.txt

test:
	venv/bin/pytest -v

lint:
	venv/bin/ruff check .
	venv/bin/black --check .

notebook:
	HOLDEM_MODE=quick venv/bin/jupyter nbconvert --to notebook --execute holdem_lab.ipynb --output holdem_lab_executed.ipynb

demo:
	venv/bin/python demo.py --bots 6

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
