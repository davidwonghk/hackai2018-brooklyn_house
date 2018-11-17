# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('data/na_filled.csv')

with open('./data/columns_categorical.txt', 'r') as f:
	cols_categorical = map(lambda x: x.rstrip(), f.readlines())

cols_categorical = list(cols_categorical)
cols_categorical += ['sale_price']
cols_numerical = [k for k in list(df) if k not in cols_categorical]

scaler = preprocessing.StandardScaler()

y_index = df.columns.get_loc("sale_price")
y_values = df.iloc[: , [y_index]]
df[cols_numerical] = scaler.fit_transform(df[cols_numerical], y_values)

df.to_csv('data/feature_scaled.csv', index=False)
