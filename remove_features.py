# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

df = pd.read_csv('../data/brooklyn/brooklyntrainset.csv')

num_cols = lambda : len(df.keys())

print("original: %i"% num_cols() )


#remove features which all entries are of the sames values
cols = list(df)
nunique = df.apply(pd.Series.nunique)
cols_same_value = nunique[nunique == 1].index
df.drop(cols_same_value, axis=1, inplace = True)
print("dropped all same values: %s"%cols_same_value)
print("num columns: %i"%num_cols())


#remove addresses/taxlot related features as we only keep:
#zipocode, block(tax block), SchoolDist, Council
#
#since the ML algorithm should be able to discover the releationship between the location and sales price
address_related = ['Address', 'address', 'OwnerName', 'lot', 'CD', 'CT2010', 'CB2010',
									 'FireComp', 'ZoneDist1', 'SplitZone', 'PolicePrct', 'HealthCent'
									 'HealthArea', 'SanitBoro', 'SanitDist', 'SanitSub', 'ZoneDist1', 'ZoneDist2']
df.drop(address_related, axis=1, inplace = True)


#Manually drop by reasons
	#sale_date: there is sale year
	#
cols_to_drop = ['sale_date']
df.drop(address_related, axis=1, inplace = True)
print("dropped columns manually: %s"%cols_to_drop)
print("num columns: %i"%num_cols())


#remove duplicate features/columns



#drop features with 80% of NaN
num_row_original = len(df)
df = df.dropna(axis='columns', thresh=int(num_row_original * 0.2) )
print("num columns: %i"%num_cols())




df.to_csv('data/features_removed.csv', index=False)
