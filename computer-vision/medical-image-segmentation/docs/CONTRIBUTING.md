# 🤝 Contributing Guide

Thank you for your interest in contributing to the Medical Image Segmentation project! This document provides guidelines and instructions for contributors.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to see if the bug has already been reported
2. **Create a new issue** with a clear title and description
3. **Include steps to reproduce** the bug
4. **Add environment details** (OS, Python version, PyTorch version, etc.)
5. **Include error messages** and stack traces

### Suggesting Enhancements

1. **Check existing issues** for similar suggestions
2. **Create a new issue** describing the enhancement
3. **Explain the use case** and benefits
4. **Provide examples** if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** from `develop`:
   ```bash
   git checkout -b feature/amazing-feature develop
Make your changes following the coding standards

Add tests for new functionality

Update documentation as needed

Ensure all tests pass

Submit a pull request to the develop branch

Development Setup
Prerequisites
Python 3.8+

Git

(Optional) CUDA-enabled GPU

Setup Steps
Fork and clone the repository:

bash
git clone https://github.com/your-username/medical-image-segmentation-unet.git
cd medical-image-segmentation-unet
Create virtual environment:

bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate    # Windows
Install dependencies:

bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
Install development dependencies:

bash
pip install black flake8 pytest pytest-cov pre-commit
Setup pre-commit hooks:

bash
pre-commit install
Coding Standards
Python Code Style
We use Black for code formatting and Flake8 for linting.

bash
# Format code
black medical_image_segmentation tests scripts

# Check linting
flake8 medical_image_segmentation tests scripts

# Run both
pre-commit run --all-files
Documentation Standards
Use Google-style docstrings

Include type hints for function signatures

Update relevant documentation when changing code

Add examples for complex functions

Testing Standards
Write tests for all new functionality

Maintain test coverage above 80%

Use descriptive test method names

Include both unit and integration tests

Branching Strategy
main: Production-ready code

develop: Development branch

feature/*: Feature development

bugfix/*: Bug fixes

release/*: Release preparation

Commit Message Convention
Use descriptive commit messages following this pattern:

text
type(scope): description

[optional body]

[optional footer]
Types:

feat: New feature

fix: Bug fix

docs: Documentation

style: Code style changes

refactor: Code refactoring

test: Test-related changes

chore: Maintenance tasks

Examples:

text
feat(models): add attention UNet architecture
fix(training): resolve memory leak in trainer
docs(api): update inference documentation
Testing
Running Tests
bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=medical_image_segmentation

# Run with specific marker
pytest -m "slow"  # Run slow tests
Writing Tests
Place tests in tests/ directory

Use descriptive test names

Test both success and failure cases

Use fixtures for common setup

Documentation
Building Documentation
bash
# Install documentation dependencies
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
Documentation Structure
API Reference: Auto-generated from docstrings

User Guide: Tutorials and how-tos

Development Guide: Contributor documentation

Deployment Guide: Production deployment instructions

Release Process
Update version in pyproject.toml and setup.py

Update CHANGELOG.md with release notes

Create release branch from develop

Run full test suite

Merge to main and tag release

Build and publish package to PyPI

Update documentation for new release

Getting Help
Documentation: Check the docs/ directory

Issues: Search and create issues on GitHub

Discussion: Use GitHub Discussions for questions

Email: Contact maintainers directly

Recognition
Contributors will be recognized in:

GitHub contributors list

Project documentation

Release notes

Thank you for contributing to making medical image segmentation more accessible and effective!

text

### `.gitignore`
Byte-compiled / optimized / DLL files
pycache/
*.py[cod]
*$py.class

C extensions
*.so

Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

PyInstaller
Usually these files are written by a python script from a template
before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

Installer logs
pip-log.txt
pip-delete-this-directory.txt

Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

Translations
*.mo
*.pot

Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

Flask stuff:
instance/
.webassets-cache

Scrapy stuff:
.scrapy

Sphinx documentation
docs/_build/

PyBuilder
.pybuilder/
target/

Jupyter Notebook
.ipynb_checkpoints

IPython
profile_default/
ipython_config.py

pyenv
For a library or package, you might want to ignore these files since the code is
intended to run in multiple environments; otherwise, check them in:
.python-version
pipenv
According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
However, in case of collaboration, if having platform-specific dependencies or dependencies
having no cross-platform support, pipenv may install dependencies that don't work, or not
install all needed dependencies.
#Pipfile.lock

poetry
Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
This is especially recommended for binary packages to ensure reproducibility, and is more
commonly ignored for libraries.
https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

pdm
Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock

pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
in version control.
https://pdm.fming.dev/#use-with-ide
.pdm.toml

PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
pypackages/

Celery stuff
celerybeat-schedule
celerybeat.pid

SageMath parsed files
*.sage.py

Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

Spyder project settings
.spyderproject
.spyproject

Rope project settings
.ropeproject

mkdocs documentation
/site

mypy
.mypy_cache/
.dmypy.json
dmypy.json

Pyre type checker
.pyre/

pytype static type analyzer
.pytype/

Cython debug symbols
cython_debug/

PyCharm
JetBrains specific template is maintained in a separate JetBrains.gitignore that can
be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
and can be added to the global gitignore or merged into this file. For a multi-project
setup, it may be better to not include this directory and instead use a per-project
.gitignore to ignore the project's .idea directory.
.idea/

vs code
.vscode/

Data files
data/raw/
data/processed/*.h5
data/processed/*.npy

Model files (large)
models/.pth
models/.pt
checkpoints/*.pth

Output files
outputs/
logs/

MLflow
mlruns/

Docker
.dockerignore

Mac
.DS_Store

Windows
Thumbs.db

text

## 📊 Additional Files

### `LICENSE`
```text
MIT License

Copyright (c) 2024 Alexander Tarasov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.