from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class MinHash:
    """
    Class to apply minhashing to a Pandas dataframe string column. Tokenization is done by Scikit-Learn's
    `CountVectorizer`.

    Args:
        n_hash_tables: nr of hash tables
        ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
        analyzer: {'word', 'char', 'char_wb'}, whether the feature should be made of word n-gram or character
            n-grams, 'char_wb' creates character n-grams only from text inside word boundaries
        **kwargs: other CountVectorizer arguments

    """
    def __init__(self, n_hash_tables: int = 10, ngram_range: Tuple[int] = (1, 1), analyzer: str = 'word',
                 **kwargs) -> None:
        self.cv = CountVectorizer(binary=True, ngram_range=ngram_range, analyzer=analyzer, **kwargs)
        self.n_hashes = n_hash_tables
        self.max_token_value = 2 ** 32 - 1
        self.next_prime = 4294967311  # `sympy.nextprime(self.max_token_value)`
        self.a = self._create_hashing_parameters()
        self.b = self._create_hashing_parameters()

    def _sparse_vectorize(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Vectorize text data in column `col_name` in Pandas dataframe `df` into sparse format.

        Args:
            df: Pandas dataframe
            col_name: columns name to be used for vectorization

        Returns:
            Pandas dataframe with a new column `sparse_vector` containing arrays with sparse vector representations of
            `col_name`

        """
        self.cv.fit(df[col_name])
        df['sparse_vector'] = self.cv.transform(df[col_name]).tolil().rows
        return df

    def _create_hashing_parameters(self) -> np.array:
        """
        Creates random integer values to be used as `a` and `b` parameters for hashing. The random values are drawn from
        the domain [0,max_token_value] and have the size `n_hash_tables`

        Returns:
            random hashing parameter values

        """
        return np.random.randint(0, self.max_token_value, self.n_hashes, dtype=int)

    def _create_minhash(self, doc: List[int]) -> np.array:
        """
        Calculate minhash values for documents `doc` represented as sparse vectors.

        Args:
            doc: sparse vector representation of a document

        Returns:
            Numpy array of size `n_hash_tables` containing minhash representations of `doc`

        """
        hashes = np.matmul(np.asarray(doc).reshape(-1, 1), self.a.reshape(1, -1))
        hashes += self.b
        hashes %= self.next_prime
        minhashes = hashes.min(axis=0)
        return minhashes

    def _create_minhash_signatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply minhashing to the column `sparse_vector` in Pandas dataframe `df` in the new column `minhash_signature`.
        In addition, one column (e.g.: 'hash_{0}') per hash table is created.

        Args:
            df: Pandas dataframe containing a column `sparse_vector`

        Returns:
            Pandas dataframe containing minhash signatures

        """
        df['minhash_signature'] = df['sparse_vector'].apply(self._create_minhash)
        df[[f'hash_{x}' for x in range(self.n_hashes)]] = df['minhash_signature'].apply(pd.Series)
        return df

    def _create_pairs(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Create pairs of rows that have at least one minhash signature in common. The column `jaccard_sim` contains
        Jaccard similarity values.

        Args:
            df: Pandas dataframe containing minhash signatures in the columns `hash_{n}` with {n} ranging from 0 to
            n_hash_tables.
            col_name: column name on which minhashing is applied

        Returns:
            Pandas dataframe containing pairs of strings with non-zero Jaccard similarity

        """
        pairs_table = pd.DataFrame()
        for h in range(self.n_hashes):
            comparison = df.merge(df[['row_number', f'hash_{h}']], on=f'hash_{h}', how='left', suffixes=('_1', '_2'))
            comparison = comparison[comparison['row_number_1'] < comparison['row_number_2']]
            comparison[f'hash_{h}'] = 1
            pairs_table = pairs_table.append(comparison[['row_number_1', 'row_number_2', f'hash_{h}']],
                                             ignore_index=True)
        pairs_table = pairs_table.fillna(0)

        pairs_table = (pairs_table.groupby(['row_number_1', 'row_number_2'], as_index=False)
                       [[f'hash_{x}' for x in range(self.n_hashes)]]
                       .sum())

        pairs_table['jaccard_sim'] = (pairs_table[[f'hash_{x}' for x in range(self.n_hashes)]].sum(axis=1) /
                                      self.n_hashes)

        pairs_table = (pairs_table
                       .merge(df[['row_number', col_name]], left_on='row_number_1', right_on='row_number')
                       .drop(columns=['row_number'])
                       .rename(columns={col_name: f'{col_name}_1'})
                       .merge(df[['row_number', col_name]], left_on='row_number_2', right_on='row_number')
                       .drop(columns=['row_number'])
                       .rename(columns={col_name: f'{col_name}_2'})
                       )

        return (pairs_table[['row_number_1', 'row_number_2', f'{col_name}_1', f'{col_name}_2', 'jaccard_sim']]
                .sort_values('jaccard_sim', ascending=False))

    def fit_predict(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Create pairs of rows in Pandas dataframe `df` in column `col_name` that have a non-zero Jaccard similarity.
        Jaccard similarities are added to the column `jaccard_sim`.

        Args:
            df: Pandas dataframe
            col_name: column name to use for matching

        Returns:
            Pandas dataframe containing pairs of (partial) matches

        """
        df_ = df[[col_name]].drop_duplicates().copy()
        if 'row_number' not in df_.columns:
            df_['row_number'] = np.arange(len(df_))
        df_ = self._sparse_vectorize(df_, col_name)
        df_ = self._create_minhash_signatures(df_)
        return self._create_pairs(df_, col_name)
