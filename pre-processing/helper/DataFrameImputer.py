# -*- coding: utf-8 -*-

from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

"""
Impute missing values.
Columns of dtype object are imputed with the most frequent value in column.
Columns of other types are imputed with mean of column.
"""
class DataFrameImputer(TransformerMixin):
	def __init__(self, missing_values=np.nan, strategy="mean"):
		self.missing_values = missing_values
		self.strategy = strategy

	def fit(self, X, y=None):
		self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], index=X.columns)
		return self

	def transform(self, X, y=None):
		return X.fillna(self.fill)
