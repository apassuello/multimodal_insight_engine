.PHONY: help install lint format type-check test test-fast clean docs setup check-deps demo-monitoring

help:
	@echo "MultiModal Insight Engine - Common Tasks"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          Create virtual environment and install dependencies"
	@echo "  make install        Install all dependencies (assumes venv exists)"
	@echo "  make install-dev    Install dev dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run flake8 linter"
	@echo "  make format         Auto-format code with black and isort"
	@echo "  make type-check     Run mypy type checker"
	@echo "  make check          Run all checks (lint, format, type-check)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests with coverage"
	@echo "  make test-fast      Run fast unit tests only (<30 sec)"
	@echo "  make test-verbose   Run tests with verbose output"
	@echo "  make test-monitoring Run monitoring system tests only"
	@echo ""
	@echo "Demos:"
	@echo "  make demo-monitoring Run training monitoring demo"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts and cache"
	@echo "  make clean-test     Remove test artifacts only"
	@echo ""
	@echo "Git:"
	@echo "  make pre-commit     Install pre-commit hooks"
	@echo ""

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements/base.txt -r requirements/dev.txt
	. venv/bin/activate && pre-commit install

install:
	pip install -r requirements/base.txt

install-dev:
	pip install -r requirements/base.txt -r requirements/dev.txt

lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --count --statistics --show-source

format:
	@echo "Running black..."
	black src/ tests/
	@echo "Running isort..."
	isort src/ tests/

type-check:
	@echo "Running mypy..."
	mypy src/ tests/ --ignore-missing-imports

check: lint type-check
	@echo "All checks passed!"

test:
	@echo "Running full test suite with coverage..."
	pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml \
		--tb=short

test-fast:
	@echo "Running fast unit tests only..."
	pytest tests/ -m "not slow" --tb=short -v

test-verbose:
	@echo "Running tests with verbose output..."
	pytest tests/ -v --tb=long

test-no-cov:
	@echo "Running tests without coverage..."
	pytest tests/ -v

test-monitoring:
	@echo "Running monitoring system tests only..."
	pytest tests/training/monitoring/ -v --tb=short

demo-monitoring:
	@echo "Running training monitoring demo..."
	python demos/ppo_monitoring_demo.py --quick
	@echo ""
	@echo "Demo complete! Check training_outputs/ for reports."
	@echo "For verbose mode: python demos/ppo_monitoring_demo.py --verbose --num_steps 30"
	@echo "Full user guide: demos/MONITORING_USER_GUIDE.md"

clean: clean-test
	@echo "Removing build artifacts..."
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Removing Python cache..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-test:
	@echo "Removing test artifacts..."
	rm -rf .coverage .coverage.* .pytest_cache htmlcov/ coverage.xml

pre-commit:
	@echo "Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed. Running on all files..."
	pre-commit run --all-files

check-deps:
	@echo "Checking for outdated dependencies..."
	pip list --outdated

.PHONY: help install lint format type-check test test-fast clean docs setup check-deps pre-commit
