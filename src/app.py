import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv('/workspace/Linear-Regression-Project/data/processed/df.csv')
X = df.drop(['charges'], axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))
param_grid = {'polynomialfeatures__degree': np.arange(4),'linearregression__fit_intercept': [True, False], 'linearregression__normalize': [True, False]}
grid = GridSearchCV(PolynomialRegression(), param_grid)
grid.fit(X_train, y_train)
tunned_model = grid.best_estimator_
tunned_model.fit(X_train,y_train)
filename = '/workspace/Linear-Regression-Project/models/linearRegression.pickle'
pickle.dump(tunned_model, open(filename, 'wb'))