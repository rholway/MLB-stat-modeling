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
    df1 = df.filter(['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','SO-pitched','E'])

    columns = ['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','SO-pitched','E']
    y = df['W'].values
    X = df1.values
    #standardize data
    standardizer = StandardScaler()
    X_std = standardizer.fit_transform(X)
    #test train split from standardized data
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, random_state=42)

    # X_train_std = standardizer.transform(X_train)
    # X_test_std = standardizer.transform(X_test)

    ridge = make_ridge(X_train, y_train)
    lasso = make_lasso(X_train, y_train)

    print(f'Ridge test R^2: {ridge.score(X_test, y_test)}')
    print(f'Lasso test R^2: {lasso.score(X_test, y_test)}')

    print(f'Ridge predict: {ridge.predict(X_std)}')
    print(f'Lasso predict: {lasso.predict(X_std)}')

    nalphas = 50
    min_alpha_exp = -1
    max_alpha_exp = 3.5
    nfeatures = 10
    coefs = np.zeros((nalphas, nfeatures))
    alphas = np.logspace(min_alpha_exp, max_alpha_exp, nalphas)
    for i, alpha in enumerate(alphas):
        #model = Pipeline([('standardize', StandardScaler()),
        #                  ('lasso', Lasso(alpha=alpha))])
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        #coefs[i] = model.steps[1][1].coef_
        coefs[i] = model.coef_

    fig, ax = plt.subplots(figsize=(10,5))
    for feature, color in zip(range(nfeatures),
                              ['r','g','b','c','m','k','y','b','r','g']):
        plt.plot(alphas, coefs[:, feature],
                 color=color,
                 label="$\\beta_{{{}}}$".format(columns[feature]))
    ax.set_xscale('log')
    ax.set_title("$\\beta$ as a function of $\\alpha$ for LASSO regression")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\\beta$")
    ax.legend(loc="right")
    plt.savefig('lasso-fig.png')
    plt.close()


    nalphas_ridge = 50
    min_alpha_exp_ridge = 0
    max_alpha_exp_ridge = 10
    coefs_ridge = np.zeros((nalphas_ridge, nfeatures))
    alphas_ridge = np.logspace(min_alpha_exp_ridge, max_alpha_exp_ridge, nalphas_ridge)
    for i, alpha in enumerate(alphas_ridge):
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        coefs_ridge[i] = model.coef_

    fig, ax = plt.subplots(figsize=(10,5))
    for feature, color in zip(range(nfeatures),
                              ['r','g','b','c','m','k','y','b','r','g']):
        plt.plot(alphas_ridge, coefs_ridge[:, feature],
                 color=color,
                 label="$\\beta_{{{}}}$".format(columns[feature]))

    ax.set_xscale('log')
    ax.set_title("$\\beta$ as a function of $\\alpha$ for Ridge regression")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\\beta$")
    ax.legend(loc="upper right")
    plt.savefig('ridge-fig.png')
    plt.close()
    # plt.show()
