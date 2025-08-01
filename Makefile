# Makefile for PoS Simulator Python

.PHONY: help install test run-examples clean lint format

help:  ## Show this help message
	@echo "PoS Simulator Python - Available Commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

test:  ## Run basic tests
	python test_basic.py

test-nakamoto:  ## Run Nakamoto coefficient tests
	python test_nakamoto.py

test-verbose:  ## Run tests with verbose output
	python -v test_basic.py

run-examples:  ## Run example simulations
	cd examples && python simple_example.py

quick-demo:  ## Run a quick demonstration
	python -c "import sys; sys.path.append('src'); from examples.simple_example import quick_demo; quick_demo()"

lint:  ## Run linting
	flake8 src/ examples/ test_basic.py --max-line-length=100 --ignore=E501,W503

format:  ## Format code with black
	black src/ examples/ test_basic.py --line-length=100

typecheck:  ## Run type checking
	mypy src/ --ignore-missing-imports

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.png" -delete
	find . -type f -name "*.jpg" -delete
	rm -rf results/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

build:  ## Build package
	python setup.py sdist bdist_wheel

install-package:  ## Install package in development mode
	pip install -e .

all: install test run-examples  ## Install, test, and run examples

jupyter:  ## Start Jupyter notebook
	jupyter notebook examples/

# Documentation targets
docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

# Performance testing
benchmark:  ## Run performance benchmarks
	python -c "import sys; sys.path.append('src'); from examples.simple_example import quick_demo; import time; start=time.time(); quick_demo(); print(f'Execution time: {time.time()-start:.2f}s')"