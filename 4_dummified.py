# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from category_encoders import HashingEncoder


df = pd.read_csv('data/na_filled.csv')


cols_categorical = [ k for k in df.keys() if df[k].dtype == 'O' ]

cols_categorical += [
		'block', 'zip_code', 'tax_class_at_sale', 'LandUse', 'AreaSource',
		'ProxCode', 'IrrLotCode', 'LotType', 'BsmtCode', 'CondoNo', 'MAPPLUTO_F'
]

#drop categorical columns
#df.drop(cols_categorical, axis=1, inplace = True)
encoder = HashingEncoder(cols=cols_categorical, n_components=64)

index_y = df.columns.get_loc("sale_price")
df = encoder.fit_transform(df, dataset.iloc[: , [index_y])


df.to_csv('data/output.csv', index=False)
