#!/bin/bash

# Exit on error
set -e

echo "Running tests with coverage..."

# Run tests with coverage and generate reports
python -m pytest \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    --cov-fail-under=40 \
    tests/ \
    --junitxml=reports/junit-report.xml

echo "Test reports generated:"
echo "- HTML coverage report: coverage_html/index.html"
echo "- XML coverage report: coverage.xml"
echo "- JUnit test report: reports/junit-report.xml" 