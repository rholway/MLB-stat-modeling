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

print(create_VIF_df(hit_cols))
print(hit_model.summary())
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
print(create_VIF_df(pitch_cols))
print(pitch_model.summary())
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
# plt.close()
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
# plt.close()
