from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import Lars, LassoLars, LassoLarsCV
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_log_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import operator


def make_lasso(X, y, cv=10):
    alphas = np.logspace(-3,3,50)
    lasso = LassoCV(alphas=alphas, cv=cv).fit(X,y)
    print(f'Lasso train R^2: {lasso.score(X,y)}, alpha: {lasso.alpha_}')
    return lasso

def make_ridge(X, y, cv=10):
    alphas = np.logspace(-3,3,50)
    ridge = RidgeCV(alphas=alphas, cv=cv).fit(X,y)
    print(f'Ridge train R^2: {ridge.score(X,y)}, alpha: {ridge.alpha_}')
    return ridge

if __name__ == '__main__':
    df = pd.read_excel('/Users/ryanholway/Documents/galvanize/capstone_I/data-trials/pitching-and-batting-all.xlsx')
    df['1B'] = df['H'] - df['HR'] - df['3B'] - df['2B']
    df1 = df.filter(['Year','Team','1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','E'])
    cubs2018df = df1.query("Year == 2018 and Team == 'Cubs'")
    cubs2018df.drop(['Year','Team'], axis = 1, inplace=True)
    print(cubs2018df)
