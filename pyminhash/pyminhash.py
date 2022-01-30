from typing import Tuple, List

import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, Tokenizer, NGram, VectorAssembler, MinHashLSH


class MinHash:
    """
    Class to apply minhashing to a Pandas dataframe string column. Tokenization is done by Scikit-Learn's
    `CountVectorizer`.

    Args:
        n_hash_tables: nr of hash tables
        ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted

    """

    def __init__(self, n_hash_tables: int = 10, ngram_range: Tuple[int] = (1, 1), row_id: str = "row_number") -> None:
        self.ngram_range = range(ngram_range[0], ngram_range[1] + 1)
        self.n_hashes = n_hash_tables
        self.row_id = row_id

    def _sparse_vectorize(self, df: DataFrame, col_name: str) -> DataFrame:
        """
        Vectorize text data in column `col_name` in Pandas dataframe `df` into sparse format.

        Args:
            df: Pandas dataframe
            col_name: columns name to be used for vectorization

        Returns:
            Pandas dataframe with a new column `sparse_vector` containing arrays with sparse vector representations of
            `col_name`

        """

        tokenizer = [Tokenizer(inputCol=col_name, outputCol="words")]
        ngrams = [
            NGram(n=i, inputCol="words", outputCol=f"{i}_grams")
            for i in self.ngram_range
        ]

        vectorizers = [
            CountVectorizer(inputCol=f"{i}_grams",
                            outputCol=f"{i}_counts")
            for i in self.ngram_range
        ]

        assembler = [VectorAssembler(
            inputCols=[f"{i}_counts" for i in self.ngram_range],
            outputCol="sparse_vector"
        )]

        sparse_vector_model = Pipeline(
            stages=tokenizer + ngrams + vectorizers + assembler)

        original_columns = df.columns

        return sparse_vector_model.fit(df).transform(df)\
            .select(original_columns + ['sparse_vector'])

    def _add_row_number(self, df):
        # add row number if does not exists
        if self.row_id not in df.columns:
            w = Window().orderBy(f.lit('A'))
            return df.withColumn(self.row_id, f.row_number().over(w))
        return df

    def fit_predict(self, df: DataFrame, col_name: str) -> DataFrame:
        """
        Create pairs of rows in Pandas dataframe `df` in column `col_name` that have a non-zero Jaccard similarity.
        Jaccard similarities are added to the column `jaccard_sim`.

        Args:
            df: Pandas dataframe
            col_name: column name to use for matching

        Returns:
            Pandas dataframe containing pairs of (partial) matches

        """
        df_ = self._add_row_number(df)
        
        # drop duplicate and keep the first row id
        df_ = df_.groupBy(col_name).agg(
            f.first(self.row_id).alias(self.row_id)
        )

        df_ = self._sparse_vectorize(df_, col_name)

        mh = MinHashLSH(inputCol="sparse_vector",
                        outputCol="hashes", numHashTables=self.n_hashes)
        model = mh.fit(df_)

        transform = model.transform(df_)

        # Create pairs of rows that have at least one minhash signature in common (1/self.n_hashes) The column `jaccard_sim` contains
        # Jaccard similarity values.

        pairs = model.approxSimilarityJoin(
            transform,
            transform,
            0.5,
            'jaccard_distance')
        pairs = pairs.withColumn('jaccard_sim', f.lit(
            1) - f.col('jaccard_distance')).drop('jaccard_distance')

        pairs = pairs.select(
            f.col(f"datasetA.{self.row_id}").alias(f"{self.row_id}_1"),
            f.col(f"datasetB.{self.row_id}").alias(f"{self.row_id}_2"),
            f.col(f"datasetA.{col_name}").alias(f"{col_name}_1"),
            f.col(f"datasetB.{col_name}").alias(f"{col_name}_2"),
            f.col("jaccard_sim"))
        return pairs
