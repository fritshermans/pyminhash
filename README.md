<!--- BADGES: START --->
[![Version](https://img.shields.io/pypi/v/pyminhash)](https://pypi.org/project/pyminhash/)
![](https://img.shields.io/github/license/fritshermans/pyminhash)
[![Downloads](https://pepy.tech/badge/pyminhash)](https://pepy.tech/project/pyminhash)
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/pyminhash?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/pyminhash?logo=anaconda&style=flat&color=orange)][#conda-forge-package]
[![Conda Recipe](https://img.shields.io/static/v1?logo=conda-forge&style=flat&color=green&label=recipe&message=pyminhash)][#conda-forge-feedstock]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=readthdocs&style=flat&color=pink&label=docs&message=pyminhash)][#docs-package]

[#pypi-package]: https://pypi.org/project/pyminhash/
[#conda-forge-package]: https://anaconda.org/conda-forge/pyminhash
[#conda-forge-feedstock]: https://github.com/conda-forge/pyminhash-feedstock
[#docs-package]: https://pyminhash.readthedocs.io/en/latest/
<!--- BADGES: END --->

# PyMinHash

MinHashing is a very efficient way of finding similar records in a dataset based on Jaccard similarity. PyMinHash
implements efficient minhashing for Pandas dataframes. See instructions below or look at the example notebook to get
started.

Developed by [Frits Hermans](https://www.linkedin.com/in/frits-hermans-data-scientist/)

## Documentation

Documentation can be found [here](https://pyminhash.readthedocs.io/en/latest/)

## Installation

### Normal installation
**Using PyPI**

```
pip install pyminhash
```

**Using conda**

```
conda install -c conda-forge pyminhash
```


### Install to contribute

Clone this Github repo and install in editable mode:

```
python -m pip install -e ".[dev]"
python setup.py develop
```

## Usage

Apply record matching to column `name` of your Pandas dataframe `df` as follows:

```python
myHasher = MinHash(n_hash_tables=10)
myHasher.fit_predict(df, 'name')
```

This will return the row pairs from `df` that have non-zero Jaccard similarity.
