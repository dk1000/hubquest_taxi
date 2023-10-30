import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Author:
    Date:
    Description:
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        # function to calculate and store some data in the Transformer (like mean, median etc)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series=None):
        # function to run all the transformations
        return X

    # no need to create fit_transform function - sklearn does everything for us





