# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error

# Importing the dataset
df = pd.read_csv('../data/pre_processed.csv')
X, y = np.split(df,[-1],axis=1)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.pipeline import Pipeline

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=1)),
	 ('model', Lasso(alpha=0.3, fit_intercept=True))   #regulization
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

#export the model
from sklearn.externals import joblib
joblib.dump(pipeline, 'save/linear1.joblib')


print('Training Score: {}'.format(pipeline.score(X_train, y_train)))
print('Test Score: {}'.format(pipeline.score(X_test, y_test)))


#evaludate the model
y_pred = pipeline.predict(X_test)
y_pred = np.array(y_pred)

plt.plot(y_test, np.zeros_like(y_test), 'y')
plt.show()