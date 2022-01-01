.. PyMinHash documentation master file, created by
   sphinx-quickstart on Sun Apr 18 11:28:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyMinHash documentation
=====================================
MinHashing is a very efficient way of finding similar records in a dataset based on Jaccard similarity. PyMinHash
implements efficient minhashing for Pandas dataframes. See instructions below or look at the example notebook to get
started.

Installation
------------

Install directly from PyPi:

``pip install pyminhash``

Usage
-----

Apply record matching to your Pandas dataframe `df` as follows:

``myHasher = MinHash(n_hash_tables=10)
myHasher.fit_predict(df, 'name')``

This will return the row pairs from `df` that have non-zero Jaccard similarity.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   Tutorial.ipynb
   api/modules





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
