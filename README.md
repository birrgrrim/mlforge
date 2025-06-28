# 🧰 mlforge

[![CI](https://github.com/birrgrrim/mlforge/actions/workflows/python.yml/badge.svg)](https://github.com/birrgrrim/mlforge/actions)
[![Release](https://img.shields.io/github/v/release/birrgrrim/mlforge)](https://github.com/birrgrrim/mlforge/releases)


**mlforge** is a lightweight Python toolkit to help data scientists and ML practitioners:

- Optimize and calibrate machine learning models
- Perform feature selection and ensemble voting parameter tuning
- Automate reproducible ML workflows by saving and reloading best features & hyperparameters
- Visualize bias, variance, and model performance diagnostics

Built for real-world projects and competitions, **mlforge** simplifies tuning, calibration, and reproducibility so you can focus on winning solutions.

---

## ✨ Features
- Greedy backward feature elimination and other selection strategies
- Grid/random search helpers and voting optimization
- Model calibration utilities
- Save & load best parameters and feature sets in JSON
- Bias/variance visualization and stats
- Modular design with optional dependencies: `xgboost`, `lightgbm`, `catboost`

---

## 📦 Installation

Install base (requires Python ≥3.8):

```bash
pip install mlforge
```

With optional extras:

```bash
pip install mlforge[xgboost,lgbm]
```

📝 Package is under active development; name or structure may change before first stable release.

---

## 🚀 Quickstart

```python
from mlforge.feature_selection import greedy_backward_elimination
from mlforge.calibration import calibrate_model

# Fit model, tune params, and reduce features
# ...
```

---

## 📜 License

Released under the [MIT License](LICENSE).

---

## 📌 Status

Alpha: Work in progress. Contributions and ideas welcome!