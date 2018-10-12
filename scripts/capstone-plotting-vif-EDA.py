import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from statsmodels.stats.outliers_influence import variance_inflation_factor

#find variance inflation factores between features
def create_VIF_df(features_list):
    #create new VIF df
    vif = pd.DataFrame()
    features_df = df[features_list]
    vif["VIF Factor"] = [variance_inflation_factor(features_df.values, i) for i in range(features_df.shape[1])]
    vif["features"] = features_df.columns
    return vif

def make_subplots(fig_num, subplot_loc, column, y, x_label, y_label='Team Wins Per Year'):
    '''
    create subplots of features against wins(y)
    fig_num: str - figure number (fig1, fig2, etc.)
    subplot_loc: int - location of subplot(221, 222, etc.)
    x_label: str - x label
    '''
    ax = fig_num.add_subplot(subplot_loc)
    ax.scatter(column, y)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    axes = plt.gca()
    m, b = np.polyfit(column, y , 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.plot(X_plot, m*X_plot + b, '-', c='r')


if __name__ == '__main__':

    #load in df
    df = pd.read_excel('/Users/ryanholway/Documents/galvanize/capstone_I/data-trials/pitching-and-batting-all.xlsx')
    df.rename(columns={'SO-allowed':'SO-pitched'}, inplace=True)
    df['1B'] = df['H'] - df['HR'] - df['3B'] - df['2B']

    df1 = df.filter(['1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','E'])
    df1_cols = list(df1.columns.values)

    new_df = df.filter(['OPS', 'ERA', 'WHIP', 'RBI', 'E'])
    new_df_cols = list(new_df.columns.values)

    #define target
    y = df['W'].values
    #define features
    X1 = df[df1_cols]
    new_X = df[new_df_cols]

    #hitting model
    og_model = sm.OLS(endog=y, exog=X1).fit()
    new_model = sm.OLS(endog=y, exog=new_X).fit()

    # print(create_VIF_df(df1_cols))
    # print(og_model.summary())

    # print(create_VIF_df(new_df_cols))
    # print(new_model.summary())

    # create subplot figure
    fig1 = plt.figure(figsize = (12, 8))
    ax1 = make_subplots(fig1, 221, new_df['OPS'], df['W'], 'OPS')
    ax2 = make_subplots(fig1, 222, new_df['ERA'], df['W'], 'ERA')
    ax3 = make_subplots(fig1, 223, new_df['WHIP'], df['W'], 'WHIP')
    ax4 = make_subplots(fig1, 224, new_df['RBI'], df['W'], 'RBI')
    # plt.show()
    plt.close()

    #create subplot figure
    fig2 = plt.figure(figsize = (12, 8))
    ax1 = make_subplots(fig2, 231, df1['2B'], df['W'], 'Doubles')
    ax2 = make_subplots(fig2, 232, df1['3B'], df['W'], 'Triples', y_label = '')
    ax3 = make_subplots(fig2, 233, df1['HR'], df['W'], 'Home Runs', y_label = '')
    ax4 = make_subplots(fig2, 234, df1['BB'], df['W'], 'Walks')
    ax5 = make_subplots(fig2, 235, df1['BB'], df['W'], 'Walks', y_label = '')
    # plt.show()
    plt.close()

    #create subplot figure
    fig3 = plt.figure(figsize = (12, 8))
    ax1 = make_subplots(fig3, 221, df1['H-allowed'], df['W'], 'H-allowed')
    ax2 = make_subplots(fig3, 222, df1['HR-allowed'], df['W'], 'HR-allowed')
    ax3 = make_subplots(fig3, 223, df1['BB-allowed'], df['W'], 'BB-allowed')
    ax4 = make_subplots(fig3, 224, df1['E'], df['W'], 'E')
    plt.close()

    #additional EDA
    #plotting stats of teams with over and under 95 wins
    Edf = df.filter(['W','1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','E'])
    Edf['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    wdf = Edf.query('W >= 95')
    ldf = Edf.query('W < 90')
    wy = wdf['W'].values
    ly = ldf['W'].values



# def split_team_data(df_col, win_split):
#     '''
#     df_col: df.Series - column of desired information
#     win_split: int -
#     '''
#
# winning_singles_mean = np.mean(wdf['1B'])
# losing_singles_mean = np.mean(ldf['1B'])
#
# winning_doubles_mean = np.mean(wdf['2B'])
# losing_doubles_mean = np.mean(ldf['2B'])
#
# winning_HR_mean = np.mean(wdf['HR'])
# losing_HR_mean = np.mean(ldf['HR'])
#
# winning_BB_mean = np.mean(wdf['BB'])
# losing_BB_mean = np.mean(ldf['BB'])
#
# winning_SO_mean = np.mean(wdf['SO'])
# losing_SO_mean = np.mean(ldf['SO'])
#
# winning_Hallowed_mean = np.mean(wdf['H-allowed'])
# losing_Hallowed_mean = np.mean(ldf['H-allowed'])
#
# winning_HRallowed_mean = np.mean(wdf['HR-allowed'])
# losing_HRallowed_mean = np.mean(ldf['HR-allowed'])
#
# winning_BBallowed_mean = np.mean(wdf['BB-allowed'])
# losing_BBallowed_mean = np.mean(ldf['BB-allowed'])
#
# winning_E_mean = np.mean(wdf['E'])
# losing_E_mean = np.mean(ldf['E'])


#
#
n_groups = 5
x1 = [winning_singles_mean, winning_doubles_mean, winning_HR_mean, winning_BB_mean, winning_SO_mean]
y1 = [losing_singles_mean, losing_doubles_mean, losing_HR_mean, losing_BB_mean, losing_SO_mean]

x2 = [winning_Hallowed_mean, winning_HR_mean, winning_BBallowed_mean, winning_E_mean]
y2 = [losing_Hallowed_mean, losing_HRallowed_mean, losing_BBallowed_mean, losing_E_mean]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, x1, bar_width, alpha=opacity, color='b', label='>= 95 Wins')
rects2 = plt.bar(index+bar_width, y1, bar_width, alpha=opacity, color='g', label='< 95 Wins')
plt.xlabel('Greater than or equal to 95 wins and less than 95 Wins')
plt.ylabel('Count')
plt.title('Over vs. Under 95 wins - Offensive')
plt.xticks(index + bar_width/2, ('1B', '2B', 'HR', 'BB', 'SO'))
plt.legend()
# plt.savefig('offensive-comparison.png')
plt.close()

fig, ax = plt.subplots()
index = np.arange(4)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, x2, bar_width, alpha=opacity, color='b', label='>= 95 Wins')
rects2 = plt.bar(index+bar_width, y2, bar_width, alpha=opacity, color='g', label='< 95 Wins')
plt.xlabel('Greater than or equal to 95 wins and less than 95 Wins')
plt.ylabel('Count')
plt.title('Over vs. Under 95 wins  - Pitching / Defense')
plt.xticks(index + bar_width/2, ('H-allowed', 'HR-allowed', 'BB-allowed', 'E'))
plt.legend()
# plt.savefig('defensive-comparison.png')
plt.close()


# # Plotting yearly statistics
#
# HR_df = df[['Year', 'HR']]
# HR_df = HR_df.groupby('Year').sum().reset_index()
# fig3 = plt.figure(figsize = (12,8))
# ax1 = fig3.add_subplot(221)
# ax1.scatter(HR_df['Year'], HR_df['HR'])
# ax1.plot(HR_df['Year'], HR_df['HR'])
# ax1.set_ylabel('Total HRs')
# ax1.set_xlabel('Year')
# ax1.set_xticks(np.arange(1998, 2019, step=5))
#
#
# SO_df = df[['Year', 'SO']]
# SO_df = SO_df.groupby('Year').sum().reset_index()
# ax3 = fig3.add_subplot(223)
# ax3.scatter(SO_df['Year'], SO_df['SO'])
# ax3.plot(SO_df['Year'], SO_df['SO'])
# ax3.set_ylabel('Total SOs')
# ax3.set_xlabel('Year')
# ax3.set_xticks(np.arange(1998, 2019, step=5))
#
#
# R_df = df[['Year', 'R']]
# R_df = R_df.groupby('Year').sum().reset_index()
# ax3 = fig3.add_subplot(222)
# ax3.scatter(R_df['Year'], R_df['R'])
# ax3.plot(R_df['Year'], R_df['R'])
# ax3.set_ylabel('Total Runs')
# ax3.set_xlabel('Year')
# ax3.set_xticks(np.arange(1998, 2019, step=5))
#
# H_df = df[['Year', 'H']]
# H_df = H_df.groupby('Year').sum().reset_index()
# ax4 = fig3.add_subplot(224)
# ax4.scatter(H_df['Year'], H_df['H'])
# ax4.plot(H_df['Year'], H_df['H'])
# ax4.set_ylabel('Total Hits')
# ax4.set_xlabel('Year')
# ax4.set_xticks(np.arange(1998, 2019, step=5))
# # plt.savefig('annual-trends.png')
# plt.close()
# # plt.show()
#
# # print(HR_df.head())
# # print(HR_df.columns)
