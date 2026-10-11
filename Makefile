.PHONY: train test lint benchmark

train:
	python train.py

test:
	python -m unittest discover -s tests -v

lint:
	ruff check .
	ruff format --check .

benchmark:
	python -m bank_marketing.benchmark
