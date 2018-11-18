# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


df = pd.read_csv('data/rows_removed.csv')


num_cols = lambda : len(df.keys())

print("original: %i"% num_cols() )



#remove addresses/taxlot related features as we only keep:
#zipocode, block(tax block), Council
#ar
#since the ML algorithm should be able to discover the releationship
#between the location and sales price, regardless the associated data
address_related = ['Address', 'address',
									 'lot', 'CD', 'BBL', 'CT2010', 'CB2010', 'Tract2010',
									 'SchoolDist', 'FireComp', 'SplitZone', 'PolicePrct', 'HealthCent',
									 'HealthArea', 'SanitBoro', 'SanitDistr', 'SanitSub',
									 'ZoneDist1', 'ZoneDist2', 'ZoneDist3', 'ZoneDist4',
									 'Overlay1', 'Overlay2', 'SPDist1', 'SPDist2', 'SPDist3',
									 'LtdHeight', 'XCoord', 'YCoord', 'ZoneMap', 'ZMCode', 'Sanborn',
									 'TaxMap', 'APPBBL',
									 'PLUTOMapID', 'FIRM07_FLA', 'PFIRM15_FL', ]
df.drop(address_related, axis=1, inplace = True)


#Manually drop by reasons
	#sale_date: there is sale year
	#OwnerName: by logic shouldn't be relvant to house price
	#tax_class: more or less the same as tax_class_at_sale
	#ZipCode: same as zip_code
	#X, Unnamed: not recommended to use these two features in your model,
	# as they are assigned manually by the data source
cols_to_drop = ['sale_date', 'OwnerName', 'tax_class', 'ZipCode', 'X', 'Unnamed: 0' ]
df.drop(cols_to_drop, axis=1, inplace = True)
print("dropped columns manually: %s"%cols_to_drop)
print("num columns: %i"%num_cols())


#remove features which all entries are of the sames values
nunique = df.apply(pd.Series.nunique)
cols_same_value = nunique[nunique == 1].index
df.drop(cols_same_value, axis=1, inplace = True)
print("dropped all same values: %s"%cols_same_value)
print("num columns: %i"%num_cols())




#remove duplicate features/columns



#drop features with 80% of NaN
num_row_original = len(df)
df = df.dropna(axis='columns', thresh=int(num_row_original * 0.2) )
print("num columns: %i"%num_cols())


df.to_csv('data/columns_removed.csv', index=False)
