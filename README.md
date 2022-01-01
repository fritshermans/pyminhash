[![Version](https://img.shields.io/pypi/v/pyminhash)](https://pypi.org/project/pyminhash/)
![](https://img.shields.io/github/license/fritshermans/pyminhash)
[![Downloads](https://pepy.tech/badge/pyminhash)](https://pepy.tech/project/pyminhash)

# PyMinHash

MinHashing is a very efficient way of finding similar records in a dataset based on Jaccard similarity. PyMinHash
implements efficient minhashing for Pandas dataframes. See instructions below or look at the example notebook to get
started.

Developed by [Frits Hermans](https://www.linkedin.com/in/frits-hermans-data-scientist/)

## Documentation

Documentation can be found [here](https://pyminhash.readthedocs.io/en/latest/)

## Installation

### Normal installation

Install directly from PyPi:

```
pip install pyminhash
```

### Install to contribute

Clone this Github repo and install in editable mode:

```
python -m pip install -e ".[dev]"
python setup.py develop
```

## Usage

Apply record matching to your Pandas dataframe `df` as follows:

```python
myHasher = MinHash(n_hash_tables=10)
myHasher.fit_predict(df, 'name')
```

This will return the row pairs from `df` that have non-zero Jaccard similarity.
