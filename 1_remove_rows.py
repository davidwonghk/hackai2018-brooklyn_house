# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys

df = pd.read_csv('../data/brooklyn/brooklyntrainset.csv')

num_rows = lambda : len(df)
print("original: %i"%num_rows())

#drop all rows with price zero
df = df[df.sale_price > 0]
print("removed sale_price=0: %i"%num_rows())


#remove duplicate rows base on given keys
primary_keys = [ "block","lot", "address","sale_date", "sale_price"]
#df.duplicated(primary_keys)
df = df.drop_duplicates(primary_keys)
print("removed duplicated: %i"%num_rows())


#drop outliers
NUM_STD = 3.5  #outside 3.5 std
df = df[np.abs(df.sale_price-df.sale_price.mean()) <= (NUM_STD*df.sale_price.std())]
print("removed outliers with %f std: %i"%(NUM_STD, num_rows()))



df.to_csv('data/rows_removed.csv', index=False)
