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
    # print(f'Lasso train R^2: {lasso.score(X,y)}, alpha: {lasso.alpha_}')
    return lasso

def make_ridge(X, y, cv=10):
    alphas = np.logspace(-3,3,50)
    ridge = RidgeCV(alphas=alphas, cv=cv).fit(X,y)
    # print(f'Ridge train R^2: {ridge.score(X,y)}, alpha: {ridge.alpha_}')
    return ridge

def rmsle(y_actual, y_predicted):
   return sqrt(mean_squared_log_error(y_actual, y_predicted))

def rmse(y_actual, y_predicted):
   return sqrt(mean_squared_error(y_actual, y_predicted))

def get_accuracy(y_actual, y_predicted, normalize=False):
    return accuracy_score(y_actual, y_predicted)

def make_lasso_plot(nalphas, min_alpha_exp, max_alpha_exp, nfeatures, X, y):
    '''
    nalphas: int - number of alphas to test on
    min_alpha_exp: int - set minumum of logspace
    max_alpha_exp: int - set maximum of logspace
    nfeatures: int - how many features to plot
    X: numpy array - features
    y: numpy array - targets
    make sure nfeatures = columns in X
    '''
    coefs = np.zeros((nalphas, nfeatures))
    alphas = np.logspace(min_alpha_exp, max_alpha_exp, nalphas)
    for i, alpha in enumerate(alphas):
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        coefs[i] = model.coef_
    fig, ax = plt.subplots(figsize=(10,5))
    for feature, color in zip(range(nfeatures),
                              ['r','g','b','c','m','k','y','b','r','g']):
        plt.plot(alphas, coefs[:, feature],
                 color=color,
                 label="$\\beta_{{{}}}$".format(columns[feature]))#Jane says: wowza formatting {{{}}}!
    ax.set_xscale('log')
    ax.set_title("$\\beta$ as a function of $\\alpha$ for LASSO regression")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\\beta$")
    ax.legend(loc="right")
    # plt.savefig()
    plt.show()

def make_ridge_plot(nalphas, min_alpha_exp, max_alpha_exp, nfeatures, X, y):
    '''
    nalphas: int - number of alphas to test on
    min_alpha_exp: int - set minumum of logspace
    max_alpha_exp: int - set maximum of logspace
    nfeatures: int - how many features to plot
    X: numpy array - features
    y: numpy array - targets
    make sure nfeatures = columns in X
    '''
    coefs = np.zeros((nalphas, nfeatures))
    alphas = np.logspace(min_alpha_exp, max_alpha_exp, nalphas)
    for i, alpha in enumerate(alphas):
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        coefs[i] = model.coef_

    fig, ax = plt.subplots(figsize=(10,5))
    for feature, color in zip(range(nfeatures),
                              ['r','g','b','c','m','k','y','b','r','g']):
                              #Jane says: Is it problematic that there are two of some of the colors in thsi list?
                              # Would it be appropriate to iterate through a for-loop of colors?
        plt.plot(alphas, coefs[:, feature],
                 color=color,
                 label="$\\beta_{{{}}}$".format(columns[feature]))

    ax.set_xscale('log')
    ax.set_title("$\\beta$ as a function of $\\alpha$ for Ridge regression")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\\beta$")
    ax.legend(loc="upper right")
    # plt.savefig()
    plt.close()

def plot_residuals(y_actual, y_predicted, x_label, y_label='Residuals', title='Plot of Residuals'):
    '''
    Scatter plot of residuals
    y_actual: numpy array - true targets
    y_predicted: numpy array - predicted targets from model
    x_label: str - name of x label
    y_label: str - name of y label
    title: str - title
    y_actual and y_predicted must be same length
    '''
    residuals = y_actual - y_predicted
    fig, ax = plt.subplots(figsize=(12,4))
    ax.scatter(y_predicted, residuals)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(0,c='r',linestyle='--')
    # plt.savefig()
    plt.close()

def make_QQ_plot(y_actual, y_predicted, title='QQ Plot'):
    residuals = y_actual - y_predicted
    sm.graphics.qqplot(residuals, line='45', fit=True)
    plt.close()



if __name__ == '__main__':
    df = pd.read_excel('/Users/ryanholway/Documents/galvanize/capstone_I/data-trials/pitching-and-batting-all.xlsx')
    df['1B'] = df['H'] - df['HR'] - df['3B'] - df['2B']
    df1 = df.filter(['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','E'])

    columns = ['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','SO-pitched','E']
    y = df['W'].values
    X = df1.values
    #standardize data
    standardizer = StandardScaler()
    X_std = standardizer.fit_transform(X)
    #test train split from standardized data
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, random_state=42)

    #train models
    ridge = make_ridge(X_train, y_train)
    lasso = make_lasso(X_train, y_train)
    #
    # get predictions
    ridge_predictions = ridge.predict(X_std)
    lasso_predictions = lasso.predict(X_std)

    # make_lasso_plot(50, -3, 3, 10, X_std, y)

    # Various print statements from the model
    # print(f'Ridge test R^2: {ridge.score(X_test, y_test)}')
    # print(f'Lasso test R^2: {lasso.score(X_test, y_test)}')

    # print(f'Ridge predict: {ridge.predict(X_std).mean()}')
    # print(f'Lasso predict: {lasso.predict(X_std).mean()}')

    # print(f'Ridge RMSLE: {rmsle(y,ridge_predictions)}')
    # print(f'Lasso RMSLE: {rmsle(y,lasso_predictions)}')
    #
    # print(f'Ridge RMSE: {rmse(y,ridge_predictions)}')
    # print(f'Lasso RMSE: {rmse(y,lasso_predictions)}')

    # create new ridge and lasso models using different features
    new_df = df.filter(['OPS', 'ERA', 'WHIP', 'RBI', 'E'])
    new_cols = ['OPS', 'ERA', 'WHIP', 'RBI', 'E']
    new_X = new_df.values
    new_X_std = standardizer.fit_transform(new_X)
    new_Xtrain, new_Xtest, new_ytrain, new_ytest = train_test_split(new_X_std, y, random_state=42)

    new_ridge = make_ridge(new_Xtrain, new_ytrain)
    new_lasso = make_lasso(new_Xtrain, new_ytrain)

    new_ridge_predictions = new_ridge.predict(new_X_std)
    new_lasso_predictions = new_lasso.predict(new_X_std)

    # Various print statements from the new model
    print(f'Ridge RMSE: {rmse(y,new_ridge_predictions)}')
    print(f'Lasso RMSE: {rmse(y,new_lasso_predictions)}')
    #
    print(f'Ridge test R^2: {new_ridge.score(new_Xtest, new_ytest)}')
    print(f'Lasso test R^2: {new_lasso.score(new_Xtest, new_ytest)}')

    df_2018_true = df[0::21].filter('W')
    arr_2018_true = df_2018_true.values
    arr_2018_predict = new_lasso_predictions[0::21]
    print(df['Team'].unique())
    my_xticks = ['Cubs' 'Brewers' 'Cardinals' 'Pirates' 'Reds' 'Braves' 'Nationals'
 'Phillies' 'Mets' 'Marlins' 'Dodgers' 'Rockies' 'D-backs' 'Giants'
 'Padres' 'Red Sox' 'Yankees' 'Rays' 'Blue Jays' 'Orioles' 'Indians'
 'Twins' 'Tigers' 'White Sox' 'Royals' 'Astros' 'Athletics' 'Mariners'
 'Rangers' 'Angles']
    index = np.arange(4)
    bar_width = 0.35

    plt.rcParams['figure.figsize'] = 10, 5
    plt.plot(arr_2018_true, c='r', label='Actual')
    plt.plot(arr_2018_predict, c='b', label='Predicted')
    plt.xlabel('Teams')
    plt.ylabel('Wins')
    plt.xticks(np.arange(30), ('Cubs', 'Brewers', 'Cardinals', 'Pirates', 'Reds',
    'Braves', 'Nationals', 'Phillies', 'Mets', 'Marlins', 'Dodgers', 'Rockies',
     'D-backs', 'Giants', 'Padres', 'Red Sox', 'Yankees', 'Rays', 'Blue Jays',
     'Orioles', 'Indians', 'Twins', 'Tigers', 'White Sox', 'Royals', 'Astros',
      'Athletics', 'Mariners', 'Rangers', 'Angles'), rotation=45)
    plt.title('Predicted Wins from Lasso Model vs. Actual Wins for 2018')
    plt.legend()
    plt.tight_layout()
    plt.savefig('2018actvspred1.png')
    # plt.show()
    plt.close()











    # print(make_QQ_plot(y, new_ridge_predictions))
