import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelectorTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, df):
    # return scipy.sparse.csr_matrix(df.loc[:,self.cols].as_matrix())
    return df.loc[:,self.cols].astype(np.float64).as_matrix()
