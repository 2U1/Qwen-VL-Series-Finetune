#!/bin/bash
# Quick script to run tests with common options

set -e

echo "=========================================="
echo "UITARS Episode Loader - Test Suite"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing..."
    pip install pytest pytest-cov
fi

# Run tests with verbose output
echo "Running all tests..."
pytest tests/ -v --tb=short

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
pytest tests/ -v --tb=line --quiet

echo ""
echo "For coverage report, run:"
echo "  pytest tests/ --cov=eval --cov-report=html"
echo "  open htmlcov/index.html"

