# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV

df = pd.read_csv('../data/encoded.csv')

y = df.sale_price



#drop the features with Lasso score < threshold
to_drop = []
for c in df.keys():
	X = df[[c]]
	score = LassoCV(cv=5, random_state=12).fit(X, y).score(X, y)
	if score < 0.0001:
		to_drop.append(c)

df.drop(to_drop, axis=1, inplace = True)

df2 =  pd.read_csv('../data/encoded.csv')

df.to_csv("../data/pre_processed.csv", index=False)