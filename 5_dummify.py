# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from category_encoders import HashingEncoder


N_COMPONENT= 128

df = pd.read_csv('data/feature_scaled.csv')

with open('./data/columns_categorical.txt', 'r') as f:
	cols_categorical = f.readlines()

#drop categorical columns
#df.drop(cols_categorical, axis=1, inplace = True)
encoder = HashingEncoder(cols=cols_categorical, n_components=N_COMPONENT)

y_index = df.columns.get_loc("sale_price")
y_values = df.iloc[: , [y_index]]
df = encoder.fit_transform(df, y_values)


df.to_csv("data/encoded_%i.csv"%N_COMPONENT, index=False)
