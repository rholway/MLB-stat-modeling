from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import Lars, LassoLars, LassoLarsCV
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_log_error, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from math import sqrt
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

def rmsle(y_actual, y_predicted):
   return sqrt(mean_squared_log_error(y_actual, y_predicted))

def rmse(y_actual, y_predicted):
   return sqrt(mean_squared_error(y_actual, y_predicted))

def get_accuracy(y_actual, y_predicted, normalize=False):
    return accuracy_score(y_actual, y_predicted)



if __name__ == '__main__':
    df = pd.read_excel('/Users/ryanholway/Documents/galvanize/capstone_I/data-trials/pitching-and-batting-all.xlsx')
    df['1B'] = df['H'] - df['HR'] - df['3B'] - df['2B']
    df1 = df.filter(['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','E'])

    wdf = df.filter(['W','1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','E'])
    wdf = wdf.query('W > 95')
    wdf_x = wdf.drop(['W'], axis=1)

    columns = ['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','SO-pitched','E']
    y = df['W'].values
    X = df1.values

    w_columns = ['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','SO-pitched','E']
    w_y = wdf['W']
    w_X = wdf_x.values
    #standardize data
    standardizer = StandardScaler()
    X_std = standardizer.fit_transform(X)
    #test train split from standardized data
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, random_state=42)

    w_X_std = standardizer.fit_transform(w_X)




    ridge = make_ridge(X_train, y_train)
    lasso = make_lasso(X_train, y_train)

    ridge_predictions = ridge.predict(X_std)
    lasso_predictions = lasso.predict(X_std)

    # print(f'Ridge test R^2: {ridge.score(X_test, y_test)}')
    # print(f'Lasso test R^2: {lasso.score(X_test, y_test)}')

    # print(f'Ridge predict: {ridge.predict(X_std).mean()}')
    # print(f'Lasso predict: {lasso.predict(X_std).mean()}')

    # not sure about these
    # print(f'Ridge predict: {ridge.predict(X_std[0:])}')
    # print(f'Lasso predict: {lasso.predict(X_std[0:])}')

    # print(f'Ridge RMSLE: {rmsle(y,ridge_predictions)}')
    # print(f'Lasso RMSLE: {rmsle(y,lasso_predictions)}')
    #
    # print(f'Ridge RMSE: {rmse(y,ridge_predictions)}')
    # print(f'Lasso RMSE: {rmse(y,lasso_predictions)}')

    # print(f'Ridge accuracy: {get_accuracy(y,rounded_ridge)}')
    # print(f'Lasso accuracy: {get_accuracy(y,rounded_lasso)}')

    print(wdf.shape)


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

    #plot residuals vs Wins
    lasso_resids = y - lasso_predictions
    ridge_resids = y - ridge_predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.scatter(lasso_predictions, lasso_resids)
    ax1.set_xlabel('Predicted Wins - Lasso')
    ax1.set_ylabel('Residuals')
    ax1.axhline(0,c='r',linestyle='--')
    ax2.scatter(ridge_predictions, ridge_resids)
    ax2.set_xlabel('Predicted Wins - Ridge')
    # ax2.set_ylabel('Residuals')
    ax2.axhline(0,c='r',linestyle='--')
    plt.savefig('residuals.png')
    plt.close()
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    sm.graphics.qqplot(lasso_resids, line='45', fit=True, ax=ax1)
    ax1.set_title('Lasso QQ')
    sm.graphics.qqplot(ridge_resids, line='45', fit=True, ax=ax2)
    ax2.set_title('Ridge QQ')
    plt.savefig('QQplots.png')
    plt.close()
    # plt.show()
