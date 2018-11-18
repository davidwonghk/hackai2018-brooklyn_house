import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


df = pd.read_csv('data/columns_removed.csv')



#find columns with missing values
cols_with_nan = list(df.columns[df.isnull().any()])
print ("columns with missing values: ")
print ( cols_with_nan )


#df.fillna(df.mean(), inplace=True)

#DataFrameImputer to impute both numerical and catcategorical data
from helper.DataFrameImputer import DataFrameImputer
imp = DataFrameImputer()
df = imp.fit_transform(df)

#If the AREA is zero, data is not available for the column.
cols_area = ['BldgArea', 'LotArea']
df.fillna({ col:df[col].mean() for col in cols_area})

#regr = linear_model.LinearRegression()
#regr.fit(X, y)


cols_with_nan_after_filled = list(df.columns[df.isnull().any()])
print ("columns with missing values (after filled): ")
print ( cols_with_nan_after_filled )


df.to_csv('data/na_filled.csv', index=False)
