# Running Tests

- [Running Tests](#running-tests)
  - [Data Preparation](#data-preparation)
  - [Environment Preparation](#environment-preparation)
  - [Running tests through pytest](#running-tests-through-pytest)

## Data Preparation

Download data from the file server, and extract files to `test/data`.

```
sh script/download_test_data.sh
```

## Environment Preparation

Install packages for test.

```
pip install -r requirements/tests.txt
```

## Running tests through pytest

Running all the tests below `test/`. It is a good way to validate whether `XRMoCap` has been correctly installed:

```
pytest test/
```

Generate a coverage for the test:

```
coverage run --source xrmocap -m pytest test/
coverage xml
coverage report -m
```
