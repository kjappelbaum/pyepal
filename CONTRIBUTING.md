# Contributing to PyPAL

If you find a bug or have a feature request, please open an issue. If you report a bug, please use the issue template.
For help, you can also use the Gitter chat.

## Development guidelines

### Commit messages

- To automatically generate the changelog and version numbers we use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0-beta.2/)

### Python code

Please install the pre-commit hooks to automatically

- format the code with [black](https://github.com/psf/black)
- sort the imports with [isort](https://pycqa.github.io/isort/)
- lint the code with [prospector](http://prospector.landscape.io/en/master/)

We use type hints, which we feel is a good way of documentation and helps us find bugs using [mypy](http://mypy-lang.org/).

### New features

Please make a new branch for the development of new features.
