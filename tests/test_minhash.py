import pytest

from pyminhash import MinHash
from pyminhash.datasets import load_data


def test__sparse_vector():
    df = load_data()
    myMinHasher = MinHash(10)
    res = myMinHasher._sparse_vectorize(df, 'name')
    assert res.columns.tolist() == ['name', 'sparse_vector']
    assert res['sparse_vector'].dtype == 'object'


def test__create_hashing_parameters():
    n_hashes = 10
    myMinHasher = MinHash(n_hash_tables=n_hashes)
    res = myMinHasher._create_hashing_parameters()
    assert len(res) == n_hashes
    assert res.dtype == 'int'
    assert min(res) >= 0
    assert min(res) <= myMinHasher.max_token_value


def test__create_minhash():
    n_hashes = 10
    myMinHasher = MinHash(n_hash_tables=n_hashes)
    doc = [59, 65, 66, 67, 118, 150, 266]
    res = myMinHasher._create_minhash(doc)
    assert len(res) == n_hashes


def test__create_minhash_signatures():
    df = load_data()
    myMinHasher = MinHash(3)
    df = myMinHasher._sparse_vectorize(df, 'name')
    df = myMinHasher._create_minhash_signatures(df)
    for col in ['hash_0', 'hash_1', 'hash_2']:
        assert col in df.columns
        assert df[col].dtype == 'int'


def test_fit_predict():
    df = load_data()
    myMinHasher = MinHash(100)
    res = myMinHasher.fit_predict(df, 'name')
    assert res.columns.tolist() == ['row_number_1', 'row_number_2', 'name_1', 'name_2', 'jaccard_sim']
    assert res['jaccard_sim'].dtype == 'float'
    assert len(res) == pytest.approx(1727, 10)
