[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-31311/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-red)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/janschleicher/bowside/graph/badge.svg?token=UqgdiVWAQi)](https://codecov.io/gh/janschleicher/bowside)
[![Python package](https://github.com/janschleicher/bowside/actions/workflows/python-package.yml/badge.svg)](https://github.com/janschleicher/bowside/actions/workflows/python-package.yml)

# bowside

Biologically-inspired Optimization of regulatory Weights for Simulation and Inference via Differential Equations (*bowside*) is a method for GRN-driven mechanistic modeling of cell differentiation.

## Installation

### Latest version from GitHub

Clone the repository:

    git clone git@github.com:janschleicher/bowside.git

Change into the directory and install with pip:

    cd bowside
    pip install -e .

## Contributing

For contributing, you should install `bowside` in development mode:

    pip install -e ".[dev]"

This will install the additional dependencies `ruff` and `pytest`, which are used for formatting and code style, and testing, respectively. Please run these before commiting new code.
