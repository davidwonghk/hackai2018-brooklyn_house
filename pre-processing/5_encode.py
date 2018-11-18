# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher

N_FEATURES = 32


df = pd.read_csv('../data/feature_scaled.csv')

with open('../data/columns_categorical.txt', 'r') as f:
	cols_categorical = [k.rstrip() for k in f.readlines()]

#print out categorical columns information
#and prepare hash encoding labels > 32
from collections import Counter

cols_to_hash = []
for c in set(cols_categorical):
	count = len(Counter(df[c]))
	print("%s: %i"%( c, count ))
	if count > N_FEATURES:
		cols_to_hash.append(c)


#use hash encoder to encode it to reduce the final number of features
#since the following features has too many labels:
	#block: 6747
	#apartment_number: 3834
cols_not_hash = [c for c in cols_categorical if c not in cols_to_hash]
print("one hot encode %s"%cols_not_hash)
df = pd.get_dummies(data=df, drop_first=True, columns=cols_not_hash)

y = df['sale_price']
for col in cols_to_hash:
	print("hash encode %s"%col)
	encoder = FeatureHasher(n_features=N_FEATURES, input_type='string')
	encoded = encoder.fit_transform([str(v) for v in df[col].values], y)
	df_encoded = pd.DataFrame(encoded.toarray(), columns=["%s_hash_%i"%(col, i) for i in range(N_FEATURES)])
	df = pd.concat([df, df_encoded], axis=1)
	df.drop(col, axis=1, inplace = True)

#move sale_price to last column for processing's sake
cols = list(df)
col_y = cols.pop(cols.index('sale_price'))
cols.append(col_y)
df = df.ix[:, cols]

df.to_csv("../data/encoded.csv", index=False)