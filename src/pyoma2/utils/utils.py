import pandas as pd


def read_from_test_data(path: str, sep: str = "\t") -> pd.DataFrame:
    """open txt file and return a df from it"""
    return pd.read_csv(path, sep=sep, header=None)
