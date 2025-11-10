.PHONY: help test lint format type-check check install verify clean docs demo-monitoring test-monitoring

# Default target
help:
	@echo "MultiModal Insight Engine - Development Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install dependencies"
	@echo "  make verify       Verify installation"
	@echo "  make dev-install  Install dev dependencies"
	@echo "  make dev-setup    Complete development setup"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run tests with coverage"
	@echo "  make test-fast    Run tests without coverage"
	@echo "  make test-verbose Run tests in verbose mode"
	@echo "  make test-file FILE=path/to/test.py"
	@echo "                    Run a specific test file"
	@echo "  make test-monitoring Run monitoring system tests only"
	@echo ""
	@echo "Demos:"
	@echo "  make demo-monitoring Run training monitoring demo"
	@echo ""
	@echo "Quality Checks:"
	@echo "  make lint         Run flake8 and mypy"
	@echo "  make format       Format code with black and isort"
	@echo "  make type-check   Run mypy type checking"
	@echo "  make check        Run all quality checks"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         View documentation index"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove build artifacts and cache"
	@echo ""

# Installation targets
install:
	pip install -r requirements.txt
	pip install -e .

dev-install:
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8 mypy
	pip install -e .

verify:
	python verify_install.py

# Test targets
test:
	python -m pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:coverage_html \
		--cov-fail-under=40 \
		-v

test-fast:
	python -m pytest tests/ -v --tb=short

test-verbose:
	python -m pytest tests/ -vv --tb=long

test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=tests/test_models.py"; \
		exit 1; \
	fi
	python -m pytest $(FILE) -v

test-monitoring:
	@echo "Running monitoring system tests only..."
	pytest tests/training/monitoring/ -v --tb=short

# Demo targets
demo-monitoring:
	@echo "Running training monitoring demo..."
	python demos/ppo_monitoring_demo.py --quick
	@echo ""
	@echo "Demo complete! Check training_outputs/ for reports."
	@echo "For verbose mode: python demos/ppo_monitoring_demo.py --verbose --num_steps 30"
	@echo "Full user guide: demos/MONITORING_USER_GUIDE.md"

# Quality check targets
lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=99 --statistics
	@echo ""
	@echo "Running mypy..."
	mypy src/ tests/ --ignore-missing-imports

format:
	@echo "Running black..."
	black src/ tests/ demos/
	@echo ""
	@echo "Running isort..."
	isort src/ tests/ demos/

type-check:
	mypy src/ tests/ --ignore-missing-imports

check: lint test
	@echo ""
	@echo "All quality checks passed!"

# Documentation
docs:
	@echo "Documentation is in docs/ directory"
	@echo "View docs/INDEX.md for navigation"

# Cleanup targets
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .coverage
	rm -rf coverage_html/ training_outputs/
	@echo "Cleanup complete"

# Watch mode (requires pytest-watch)
watch:
	ptw tests/ -- --cov=src -v

# Performance profiling
profile:
	python -m cProfile -o profile.stats demo_constitutional_ai.py
	@echo "Profile saved to profile.stats"
	@echo "View with: python -m pstats profile.stats"

# Development setup
dev-setup: install dev-install verify
	@echo ""
	@echo "Development environment ready!"
	@echo "Run 'make check' to verify code quality"
	@echo "Run 'make test' to run tests"

# Show test coverage
coverage:
	python -m pytest tests/ --cov=src --cov-report=html
	@echo ""
	@echo "Coverage report generated: coverage_html/index.html"
	@echo "Open in browser to view detailed coverage"

# Run a specific test function
test-function:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-function TEST=tests/test_models.py::test_transformer"; \
		exit 1; \
	fi
	python -m pytest $(TEST) -v

# Common development workflow shortcuts
quick-check:
	python -m pytest tests/ -q --tb=line

# Pre-commit check (for use in git hooks)
pre-commit-check: type-check test-fast
	@echo "Pre-commit checks passed!"
