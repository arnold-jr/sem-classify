import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelectorTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, df):
    return df.loc[:,self.cols].applymap(
      lambda x: x if not np.isnan(x) else np.exp(-3*random.random()))\
      .astype(np.float64).as_matrix()
