import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.plotly as py
# from plotly.graph_objs import *
from sklearn.preprocessing import scale
from statsmodels.stats.outliers_influence import variance_inflation_factor

#read df into Pandas
df = pd.read_excel('/Users/ryanholway/Documents/galvanize/capstone_I/data-trials/pitching-and-batting-all.xlsx')
df['1B'] = df['H'] - df['HR'] - df['3B'] - df['2B']
df1 = df.filter(['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed', 'E'])
df2 = df.filter(['R/G', 'OPS', 'Fld%', 'ERA', 'WHIP'])
df3 = df.filter(['R/G', 'G', 'PA', 'AB', 'R', '2B',
       '3B', 'HR', 'RBI', 'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS',
       'E', 'DP', 'Fld%', 'RA/G', 'ERA', 'H-allowed', 'R-allowed', 'ER-allowed', 'HR-allowed', 'BB-allowed',
       'SO-allowed', 'WHIP','1B'])

#make average age column
# df['AvAge'] = (df['BatAge'] + df['PAge']) / 2

#start dropping columns
# df_h = df.drop(['Team','L','R','Year', '3B','CS', 'Finish',
# 'G', 'PA', 'AB', 'BatAge', 'CG', 'tSho', 'SV', 'IP', 'PAge', 'E',
# 'SO9-pitching', 'HR9-allowed', 'ER-allowed', 'R-allowed', 'BB-allowed', 'RBI',
# 'H', 'BA', 'H-allowed', 'RA/G', 'R/G','OBP', 'SLG', 'OPS',
# 'Fld%', 'SO-allowed', 'WHIP', 'ERA', 'HR-allowed'], axis=1)




# values_of_col_list_h = list(df_h.columns.values)
#define target
# y = df['W']
#define features
# X_h = df[values_of_col_list_h]
#standardize X and y
# y_scaled = scale(y)
# X_scaled = scale(X)


# baseball_model_hitting = sm.OLS(endog=y, exog=X_h).fit()
# baseball_model_scaled = sm.OLS(endog=y_scaled, exog=X_scaled).fit()


# pd.plotting.scatter_matrix(df1)
# plt.show()

def create_VIF_df(features_list):
    #create new VIF df
    vif = pd.DataFrame()
    features_df = df[features_list]
    vif["VIF Factor"] = [variance_inflation_factor(features_df.values, i) for i in range(features_df.shape[1])]
    vif["features"] = features_df.columns
    return vif

# print(create_VIF_df(values_of_col_list_h))
# print(baseball_model_hitting.summary())
# print (len(values_of_col_list_h))

#second df with minimul features, adding ones on
# df2 = df.filter(['W', 'RpG'], axis = 1)
# values_of_col_list2 = list(df2.columns.values)
# X2 = df[values_of_col_list2]
# y2 = df['W']
# model2 = sm.OLS(endog=y2, exog=X2).fit()
# print(create_VIF_df(values_of_col_list2))
# print(model2.summary())

# df_p = df.filter(['SO9-pitching', 'BB-allowed', 'H-allowed',
# 'SO-allowed', 'WHIP','ERA', 'HR-allowed'])
#
# values_of_col_list_p = list(df_p.columns.values)
#define target
# y = df['W']
#define features
# X_p = df[values_of_col_list_p]
# baseball_model_pitching = sm.OLS(endog=y, exog=X_p).fit()

# print(create_VIF_df(values_of_col_list_p))
# print(baseball_model_pitching.summary())

col_list_1 = list(df1.columns.values)
col_list_2 = list(df2.columns.values)
col_list_3 = list(df3.columns.values)
y = df['W']
X1 = df1[col_list_1]
X2 = df2[col_list_2]
new_model_1 = sm.OLS(endog=y, exog=X1).fit()
new_model_2 = sm.OLS(endog=y, exog=X2).fit()
# print(create_VIF_df(col_list_1))
print(df.columns)
# print(df3.columns)
