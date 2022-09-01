# Running Tests

- [Data Preparation](#data-preparation)
- [Environment Preparation](#environment-preparation)
- [Running tests through pytest](#running-tests-through-pytest)

## Data Preparation

Download data from the file server, and extract files to `tests/data`.

```
sh scripts/download_test_data.sh
```

Download weights from Internet, and extract files to `weight`.

```
sh scripts/download_weight.sh
```

## Environment Preparation

Install packages for test.

```
pip install -r requirements/test.txt
```

## Running tests through pytest

Running all the tests below `test/`. It is a good way to validate whether `XRMoCap` has been correctly installed:

```
pytest tests/
```

Or generate a coverage when testing:

```
coverage run --source xrmocap -m pytest tests/
coverage xml
coverage report -m
```

Or starts a CPU-only test on a GPU machine:

```
export CUDA_VISIBLE_DEVICES=-1
pytest tests/
```
