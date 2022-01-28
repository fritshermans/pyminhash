import pytest
import os
from pkg_resources import resource_filename
from pyspark.sql import SparkSession

# inspired from https://github.com/malexer/pytest-spark/blob/0152b555eb532710fd5bd212bd95134f9342e22f/pytest_spark/__init__.py#L101


@pytest.fixture
def create_dataframe(spark):
    def _create_dataframe(data, schema, table=None):
        df = spark.createDataFrame(
            spark.sparkContext.parallelize(data),
            schema
        )
        if table:
            df.createOrReplaceTempView(table)
        return df
    return _create_dataframe


@pytest.fixture
def load_sample_spark_df(spark):

    def _load_sample_spark_df(file_name='stoxx50_extended_with_id.csv',
                              format=None,
                              sep=';',
                              header=True,
                              encoding="UTF-8"
                              ):
        file_path = resource_filename(
            'pyminhash', os.path.join('data', file_name))
        if not format:
            if file_path.endswith('.csv'):
                format = 'csv'
            elif file_path.endswith('.xlsx'):
                format = 'com.crealytics.spark.excel'
            else:
                format = 'orc'

        print(*f'Loading {format} from {file_path} ... ')

        if format == 'csv' or format == 'com.crealytics.spark.excel':
            result = spark.read.format(format).option("encoding", encoding).option(
                "inferSchema", "true").option("header", header).option("sep", sep).load(file_path)
        else:
            result = spark.read.format(format).load(file_path)

        return result

    return _load_sample_spark_df


def reduce_logging(_spark):
    """Reduce logging in SparkContext instance."""

    logger = _spark.sparkContext._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)


@pytest.fixture()
def spark():
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("plain_spark") \
        .config('spark.sql.shuffle.partitions', 1) \
        .config('spark.default.parallelism', 1) \
        .config('spark.rdd.compress', False) \
        .config('spark.shuffle.compress', False) \
        .config('spark.shuffle.compress', False) \
        .getOrCreate()
    reduce_logging(spark)
    yield spark
    spark.stop()
