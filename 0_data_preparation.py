# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import Counter

df = pd.read_csv('../data/brooklyn/brooklyntrainset.csv')

#generate a summary file
df.describe().to_csv("summary.csv")

df_size = lambda : (len(df), len(df.keys()))

#drop features with 80% of nan
print("original: %i"%df_size)

num_cols_original = df_size()[1]
df = df.dropna(axis='columns', thresh=int(num_col_original * 0.8) )
print("after dropping entries missing values (NA’s) is higher than 80% : %i", df_size() )

