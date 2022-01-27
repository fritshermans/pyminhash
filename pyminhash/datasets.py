import os
from pkg_resources import resource_filename

import pandas as pd


def load_data() -> pd.DataFrame:
    """
    Load example dataset containing various name ana address descriptions of Stoxx50 companies.

    Returns:
        Pandas dataframe containing single column 'name'

    """
    file_path = resource_filename('pyminhash', os.path.join('data', 'stoxx50_extended_with_id.csv'))
    df = pd.read_csv(file_path)
    return df[['name']]
