# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


df = pd.read_csv('data/na_filled.csv')


cols_categorical = [ k for k in df.keys() if df[k].dtype == 'O' ]

cols_categorical += [
		'block', 'zip_code', 'tax_class_at_sale', 'LandUse', 'AreaSource',
		'ProxCode', 'IrrLotCode', 'LotType', 'BsmtCode', 'CondoNo', 'MAPPLUTO_F'
]

#drop categorical columns
df.drop(cols_categorical, axis=1, inplace = True)
df.to_csv('data/toy_data.csv', index=False)
