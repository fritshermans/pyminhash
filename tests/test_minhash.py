import pandas as pd
import pytest

from pyminhash import MinHash
from pyminhash.datasets import load_data


def test__sparse_vector():
    df = load_data()
    myMinHasher = MinHash(10)
    res = myMinHasher._sparse_vectorize(df, 'name')
    assert res.columns.tolist() == ['name', 'sparse_vector']
    assert res['sparse_vector'].dtype == 'object'


def test__sparse_vector_zero_vectors():
    df = pd.DataFrame(
        data=[['george d'], ['andy t'], ['greg b'], ['ret'], ['pam'], ['kos'], ['andy'],
              ['pamela'], ['pamla'], ['kis'], ['paul'], ['paul d'],
              ['geirge d'], ['ndy t'], ['greg'], ['retos'], ['pipo'], ['konstas'], ['grig'], ['gre'], ['k i']],
        columns=['name'])
    myMinHasher = MinHash(10)
    res = myMinHasher._sparse_vectorize(df, 'name')
    assert (res['sparse_vector'].str.len() > 0).all()


def test__create_hashing_parameters():
    n_hashes = 10
    myMinHasher = MinHash(n_hash_tables=n_hashes)
    res = myMinHasher._create_hashing_parameters()
    assert len(res) == n_hashes
    assert res.dtype == 'int64'
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
        assert df[col].dtype == 'int64'


def test_fit_predict():
    df = load_data()
    myMinHasher = MinHash(10)
    res = myMinHasher.fit_predict(df, 'name')
    assert res.columns.tolist() == ['row_number_1', 'row_number_2', 'name_1', 'name_2', 'jaccard_sim']
    assert res['jaccard_sim'].dtype == 'float'


def test_fit_predict_accuracy():
    def jaccard(x, y):
        x_tokens = set(x.split())
        y_tokens = set(y.split())
        return len(x_tokens.intersection(y_tokens)) / len(x_tokens.union(y_tokens))

    df = load_data()
    myMinHasher = MinHash(1000)
    res = myMinHasher.fit_predict(df, 'name')

    assert len(res) == 1727
    res['jaccard_real'] = res.apply(lambda row: jaccard(row['name_1'], row['name_2']), axis=1)
    res['diff'] = res['jaccard_real'] - res['jaccard_sim']
    assert abs(res['diff'].mean()) < 0.02
    assert res['diff'].std() < 0.1
