.PHONY: all clean pytest coverage flake8 black mypy isort

CMD:=poetry run
PYMODULE:=src
TESTS:=tests

# Run all the checks which do not change files
all: mypy pytest flake8

# Run the unit tests using `pytest`
pytest:
	$(CMD) pytest $(PYMODULE) $(TESTS)

# Generate a unit test coverage report using `pytest-cov`
coverage:
	$(CMD) pytest --cov=$(PYMODULE) $(TESTS) --cov-report html

# Lint the code using `flake8`
flake8:
	$(CMD) flake8 $(PYMODULE) $(TESTS)

# Automatically format the code using `black`
black:
	$(CMD) black $(PYMODULE) $(TESTS)

# Perform static type checking using `mypy`
mypy:
	$(CMD) mypy $(PYMODULE) $(TESTS)

# Order the imports using `isort`
isort:
	$(CMD) isort $(PYMODULE) $(TESTS)
