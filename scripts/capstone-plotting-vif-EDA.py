import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from statsmodels.stats.outliers_influence import variance_inflation_factor


# HITTING MODEL
#load in df
df = pd.read_excel('/Users/ryanholway/Documents/galvanize/capstone_I/data-trials/pitching-and-batting-all.xlsx')
df.rename(columns={'SO-allowed':'SO-pitched'}, inplace=True)


#create hitting df
hit_df = df.filter(['2B', '3B', 'HR', 'BB','SO'])

#hitting column list
hit_cols = list(hit_df.columns.values)

#define target
y = df['W'].values
#define features
X1 = df[hit_cols]

#hitting model
hit_model = sm.OLS(endog=y, exog=X1).fit()

#find variance inflation factores between features
def create_VIF_df(features_list):
    #create new VIF df
    vif = pd.DataFrame()
    features_df = df[features_list]
    vif["VIF Factor"] = [variance_inflation_factor(features_df.values, i)
    for i in range(features_df.shape[1])]
    vif["features"] = features_df.columns
    return vif

# print(create_VIF_df(hit_cols))
# print(hit_model.summary())
# ============================================================================
# PITCHING MODEL
# pitch_df = df.filter(['RA/G', 'ERA', 'CG', 'tSho', 'SV',
#        'IP', 'H-allowed', 'R-allowed', 'ER-allowed', 'HR-allowed',
#        'BB-allowed', 'SO-pitched', 'WHIP', 'SO9-pitching', 'HR9-allowed'])

pitch_df = df.filter(['H-allowed', 'HR-allowed', 'BB-allowed', 'SO-pitched'])

#pitching column list
pitch_cols = list(pitch_df.columns.values)

# define target
y = df['W'].values
#define features
X2 = df[pitch_cols]

#hitting model
pitch_model = sm.OLS(endog=y, exog=X2).fit()

#find variance inflaction factors between pitching features
# print(create_VIF_df(pitch_cols))
# print(pitch_model.summary())
# ============================================================================
# HITTING PLOTTING
fig = plt.figure(figsize = (12,8))
ax1 = fig.add_subplot(231)
ax1.scatter(hit_df['2B'], y)
ax1.set_ylabel('Team Wins per Year')
ax1.set_xlabel('Team Doubles per Year')
axes = plt.gca()
m, b = np.polyfit(hit_df['2B'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')

ax2 = fig.add_subplot(232)
ax2.scatter(hit_df['3B'], y)
# ax2.set_ylabel('Team Wins per Year')
ax2.set_xlabel('Team Triples per Year')
axes = plt.gca()
m, b = np.polyfit(hit_df['3B'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')

ax3 = fig.add_subplot(233)
ax3.scatter(hit_df['HR'], y)
# ax2.set_ylabel('Team Wins per Year')
ax3.set_xlabel('Team Home Runs per Year')
axes = plt.gca()
m, b = np.polyfit(hit_df['HR'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')

ax4 = fig.add_subplot(234)
ax4.scatter(hit_df['BB'], y)
ax4.set_ylabel('Team Wins per Year')
ax4.set_xlabel('Team Walks per Year')
axes = plt.gca()
m, b = np.polyfit(hit_df['BB'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')

ax5 = fig.add_subplot(235)
ax5.scatter(hit_df['SO'], y)
# ax5.set_ylabel('Team Wins per Year')
ax5.set_xlabel('Team Strikeouts per Year')
axes = plt.gca()
m, b = np.polyfit(hit_df['SO'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')

# plt.savefig('hit-stats-fig')
plt.close()
# ============================================================================
# PITCHING PLOTTING
fig2 = plt.figure(figsize = (12,8))
ax1 = fig2.add_subplot(221)
ax1.scatter(pitch_df['H-allowed'], y)
ax1.set_ylabel('Team Wins per Year')
ax1.set_xlabel('Team Hits Allowed per Year')
axes = plt.gca()
m, b = np.polyfit(pitch_df['H-allowed'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')

ax2 = fig2.add_subplot(222)
ax2.scatter(pitch_df['HR-allowed'], y)
# ax2.set_ylabel('Team Wins per Year')
ax2.set_xlabel('Team Home Runs Allowed per Year')
axes = plt.gca()
m, b = np.polyfit(pitch_df['HR-allowed'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')

ax3 = fig2.add_subplot(223)
ax3.scatter(pitch_df['BB-allowed'], y)
ax3.set_ylabel('Team Wins per Year')
ax3.set_xlabel('Team Walks Allowed per Year')
axes = plt.gca()
m, b = np.polyfit(pitch_df['BB-allowed'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')

ax4 = fig2.add_subplot(224)
ax4.scatter(pitch_df['SO-pitched'], y)
# ax4.set_ylabel('Team Wins per Year')
ax4.set_xlabel('Team Strikeouts Pitched per Year')
axes = plt.gca()
m, b = np.polyfit(pitch_df['SO-pitched'], y, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r')


# plt.savefig('pitch-stats-fig')
plt.close()

# ============================================================================
#additional EDA
Edf = df.filter(['W','1B', '2B', '3B', 'HR', 'BB','SO','H-allowed','HR-allowed','BB-allowed','E'])
Edf['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
wdf = Edf.query('W >= 95')
ldf = Edf.query('W < 90')
wy = wdf['W'].values
ly = ldf['W'].values

winning_singles_mean = np.mean(wdf['1B'])
losing_singles_mean = np.mean(ldf['1B'])

winning_doubles_mean = np.mean(wdf['2B'])
losing_doubles_mean = np.mean(ldf['2B'])

winning_HR_mean = np.mean(wdf['HR'])
losing_HR_mean = np.mean(ldf['HR'])

winning_BB_mean = np.mean(wdf['BB'])
losing_BB_mean = np.mean(ldf['BB'])

winning_SO_mean = np.mean(wdf['SO'])
losing_SO_mean = np.mean(ldf['SO'])

winning_Hallowed_mean = np.mean(wdf['H-allowed'])
losing_Hallowed_mean = np.mean(ldf['H-allowed'])

winning_HRallowed_mean = np.mean(wdf['HR-allowed'])
losing_HRallowed_mean = np.mean(ldf['HR-allowed'])

winning_BBallowed_mean = np.mean(wdf['BB-allowed'])
losing_BBallowed_mean = np.mean(ldf['BB-allowed'])

winning_E_mean = np.mean(wdf['E'])
losing_E_mean = np.mean(ldf['E'])


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
plt.title('Winning Teams vs. Losing Teams - Offensive')
plt.xticks(index + bar_width/2, ('1B', '2B', 'HR', 'BB', 'SO'))
plt.legend()
plt.savefig('offensive-comparison.png')
plt.close()

fig, ax = plt.subplots()
index = np.arange(4)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, x2, bar_width, alpha=opacity, color='b', label='>= 95 Wins')
rects2 = plt.bar(index+bar_width, y2, bar_width, alpha=opacity, color='g', label='< 95 Wins')
plt.xlabel('Greater than or equal to 95 wins and less than 95 Wins')
plt.ylabel('Count')
plt.title('Winning Teams vs. Losing Teams - Pitching / Defense')
plt.xticks(index + bar_width/2, ('H-allowed', 'HR-allowed', 'BB-allowed', 'E'))
plt.legend()
plt.savefig('defensive-comparison.png')
plt.close()

HR_df = df[['Year', 'HR']]
# HR_df = HR_df.groupby('Year').agg({'HR':'sum'})
HR_df = HR_df.groupby('Year').sum().reset_index()


# df_grouped1 =  dftest.groupby(['A','Amt']).size().rename('count').reset_index()

fig3 = plt.figure(figsize = (12,8))
ax1 = fig3.add_subplot(221)
ax1.scatter(HR_df['Year'], HR_df['HR'])
ax1.set_ylabel('Total HRs')
ax1.set_xlabel('Year')
plt.show()

# print(HR_df.head())
# print(HR_df.columns)
