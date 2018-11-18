# Data Preprocessing

## to execute
```
use python3 to execute the python scripts in this folder one by one by one order by the file name
```

-----

## clean rows
- remove rows with sale_price == 0
- remove duplicated rows by the key ["block","lot", "address","sale_date", "sale_price"]
- remove outliers rows over the range of +-3.5 standard deviation
- remove all location related features except zip code, block(tax block), Council

## clean features
- remove the features  following features by the reason
```
sale_date: sale year features is a better option
OwnerName: not related to house price
tax_class: same as tax_class_at_sale
ZipCode: same as zip_code
X, Unnamed: not recommended to use these two features in your model, as they are assigned manually by the data source
```
- remove all features which all entries are of the sames values
- remove features which all entries are of the sames values
- drop features with 80% of missing values

## fill missing and feature scaling
- fill missing values with the means of the columns
- feature scaling

## Encode categorical features
- for categorical features with more than 32 labels, use hash encoding, otherwise one-hot encoding
- drop features with Lasso score < 0.0001
