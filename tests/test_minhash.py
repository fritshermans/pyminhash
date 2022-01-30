import pytest

from pyminhash import MinHash
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType


def test__sparse_vector(load_sample_spark_df):
    df = load_sample_spark_df()
    assert set(df.columns) == set(['id', 'name'])
    myMinHasher = MinHash(10)
    res = myMinHasher._sparse_vectorize(df, 'name')
    assert set(res.columns) == set(['id', 'name', 'sparse_vector'])
    assert dict(res.dtypes)['sparse_vector'] == 'vector'


def test_fit_predict(load_sample_spark_df):
    df = load_sample_spark_df()
    myMinHasher = MinHash(10)
    res = myMinHasher.fit_predict(df, 'name')
    assert res.columns == ['row_number_1',
                           'row_number_2', 'name_1', 'name_2', 'jaccard_sim']
    assert dict(res.dtypes)['jaccard_sim'] == 'double'


def test_fit_predict_accuracy(load_sample_spark_df):
    def jaccard(x, y):
        x_tokens = set(x.split())
        y_tokens = set(y.split())
        return len(x_tokens.intersection(y_tokens)) / len(x_tokens.union(y_tokens))

    df = load_sample_spark_df()
    myMinHasher = MinHash(100)
    res = myMinHasher.fit_predict(df, 'name')

    jaccard_udf = udf(jaccard, DoubleType())

    res = res.withColumn('jaccard_real', jaccard_udf('name_1', 'name_2'))
    res = res.withColumn('diff', col('jaccard_real') - col('jaccard_sim'))

    mean = abs(res.agg({'diff': 'mean'}).collect()[0][0])
    stddev = res.agg({'diff': 'stddev'}).collect()[0][0]
    print(f" mean={mean}, stddev={stddev}")
    assert mean < 0.02
    assert stddev < 0.1
    assert res.count() == 308
