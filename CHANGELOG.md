# 📄 Changelog

All notable changes to this project will be documented here.

## [0.2.0] - 2025-07-01
### Added
- **Autotune** wrappers method for automatic hyperparameter tuning and feature selection.
  - Supports **two feature selection strategies**:
     - `"greedy_backward"`: Iteratively removes features that do not improve cross-validation accuracy.
     - `"none"`: Disables feature selection, tuning only hyperparameters.
  - Supports **hyperparameter tuning strategies**, with extensible dispatch to add more in the future.
- Model wrappers allow **optional hyperparameters parameter** during initialization, simplifying usage.
- Updated **README** to reflect current implemented features and future roadmap.
- Generated **Sphinx documentation** published on GitHub Pages for easy browsing.

### Documentation
- Added API reference docs generated with Sphinx.
- Set up GitHub Pages site with working styling and navigation.

## [0.1.1] - 2025-06-29
### Added
- Model wrappers:
  - RandomForestModelWrapper
  - XGBoostModelWrapper
  - LightGBMModelWrapper
- JSON dump/load support (to_json, from_json)

## [0.1.0] - 2025-06-28
### Added
- Initial project skeleton with packaging files
- README.md and MIT License
- pyproject.toml with optional dependencies
- Empty Python package `mlforge` with `__init__.py`
- Starter CHANGELOG.md