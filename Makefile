# Makefile for mltune

# Python executable & virtualenv folder
PYTHON=python3.12
VENV=.venv

.PHONY: help venv activate install-dev install build clean test coverage docs html publish tag

help:
	@echo "Makefile commands:"
	@echo "  make venv            - Create virtualenv in .venv"
	@echo "  make activate        - Show activate command"
	@echo "  make install         - Install package in editable mode"
	@echo "  make install-dev     - Install dev dependencies"
	@echo "  make build           - Build dist (sdist and wheel)"
	@echo "  make clean           - Remove build artifacts"
	@echo "  make test            - Run unit tests with pytest"
	@echo "  make coverage        - Run tests with coverage report"
	@echo "  make docs            - Build docs to docs/_build/html"
	@echo "  make html            - Open local docs in browser"
	@echo "  make publish        - Upload to PyPI using twine"
	@echo "  make tag VERSION=x.y.z - Create git tag (e.g., make tag VERSION=0.2.0)"

venv:
	$(PYTHON) -m venv $(VENV)

activate:
	@echo "source $(VENV)/bin/activate"

install:
	$(VENV)/bin/pip install -e .

install-dev:
	$(VENV)/bin/pip install -e .[dev,docs,viz,xgboost,lgbm,build]

build:
	$(VENV)/bin/python -m build

clean:
	rm -rf dist/ build/ *.egg-info .coverage* htmlcov/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -r {} +

test:
	$(VENV)/bin/pytest tests -v

coverage:
	$(VENV)/bin/pytest --cov=mltune --cov-report=html tests -v

docs:
	@echo "📦 Building Sphinx docs..."
	$(MAKE) -C sphinx html
	@touch docs/.nojekyll
	@echo "✅ Docs built into docs/"

cleandocs:
	@echo "🧹 Cleaning docs/..."
	find docs -mindepth 1 ! -name '.nojekyll' -exec rm -rf {} +
	@echo "✅ docs/ cleaned"

docs:
	$(VENV)/bin/sphinx-build -b html sphinx/source docs

html:
	open docs/index.html

publish: clean build
	$(VENV)/bin/twine upload dist/*

tag:
	git tag v$(VERSION)
	git push origin v$(VERSION)
