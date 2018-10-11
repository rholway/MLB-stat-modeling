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

#QQ subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
# sm.graphics.qqplot(lasso_resids, line='45', fit=True, ax=ax1)
# ax1.set_title('Lasso QQ')
# sm.graphics.qqplot(ridge_resids, line='45', fit=True, ax=ax2)
# ax2.set_title('Ridge QQ')
# # plt.savefig('QQplots.png')
# plt.close()

#create hitting df
hit_df = df.filter(['2B', '3B', 'HR', 'BB','SO'])

#hitting column list
hit_cols = list(hit_df.columns.values)

pitch_df = df.filter(['H-allowed', 'HR-allowed', 'BB-allowed', 'SO-pitched'])

#pitching column list
pitch_cols = list(pitch_df.columns.values)

# define target
y = df['W'].values
#define features
X2 = df[pitch_cols]

#hitting model
pitch_model = sm.OLS(endog=y, exog=X2).fit()



# ============================================================================
# HITTING PLOTTING





# fig = plt.figure(figsize = (12,8))
# ax1 = fig.add_subplot(231)
# ax1.scatter(y, df1['2B'])
# ax1.set_ylabel('Team Doubles per Year')
# ax1.set_xlabel('Team Wins per Year')
# axes = plt.gca()
# m, b = np.polyfit(y,df1['2B'], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')
#
#
# ax2 = fig.add_subplot(232)
# ax2.scatter(y, df1['3B'])
# ax2.set_ylabel('Team Triples per Year')
# ax2.set_xlabel('Team Wins per Year')
# axes = plt.gca()
# m, b = np.polyfit(y, df1['3B'],  1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')
# plt.show()
#
# ax3 = fig.add_subplot(233)
# ax3.scatter(hit_df['HR'], y)
# # ax2.set_ylabel('Team Wins per Year')
# ax3.set_xlabel('Team Home Runs per Year')
# axes = plt.gca()
# m, b = np.polyfit(hit_df['HR'], y, 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')
#
# ax4 = fig.add_subplot(234)
# ax4.scatter(hit_df['BB'], y)
# ax4.set_ylabel('Team Wins per Year')
# ax4.set_xlabel('Team Walks per Year')
# axes = plt.gca()
# m, b = np.polyfit(hit_df['BB'], y, 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')
#
# ax5 = fig.add_subplot(235)
# ax5.scatter(hit_df['SO'], y)
# # ax5.set_ylabel('Team Wins per Year')
# ax5.set_xlabel('Team Strikeouts per Year')
# axes = plt.gca()
# m, b = np.polyfit(hit_df['SO'], y, 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')
#
# # plt.savefig('hit-stats-fig')
# plt.close()



# # ============================================================================
# # PITCHING PLOTTING
# fig2 = plt.figure(figsize = (12,8))
# ax1 = fig2.add_subplot(221)
# ax1.scatter(pitch_df['H-allowed'], y)
# ax1.set_ylabel('Team Wins per Year')
# ax1.set_xlabel('Team Hits Allowed per Year')
# axes = plt.gca()
# m, b = np.polyfit(pitch_df['H-allowed'], y, 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')
#
# ax2 = fig2.add_subplot(222)
# ax2.scatter(pitch_df['HR-allowed'], y)
# # ax2.set_ylabel('Team Wins per Year')
# ax2.set_xlabel('Team Home Runs Allowed per Year')
# axes = plt.gca()
# m, b = np.polyfit(pitch_df['HR-allowed'], y, 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')
#
# ax3 = fig2.add_subplot(223)
# ax3.scatter(pitch_df['BB-allowed'], y)
# ax3.set_ylabel('Team Wins per Year')
# ax3.set_xlabel('Team Walks Allowed per Year')
# axes = plt.gca()
# m, b = np.polyfit(pitch_df['BB-allowed'], y, 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')
#
# ax4 = fig2.add_subplot(224)
# ax4.scatter(pitch_df['SO-pitched'], y)
# # ax4.set_ylabel('Team Wins per Year')
# ax4.set_xlabel('Team Strikeouts Pitched per Year')
# axes = plt.gca()
# m, b = np.polyfit(pitch_df['SO-pitched'], y, 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-', c='r')

#
# # plt.savefig('pitch-stats-fig')
# plt.close()
