# Capstone I

# Analyzing MLB Statistics to Predict Wins

### - 30 teams in MLB since 1998
### - Took team statistics from 1998 until 2018
### - Total of 630 teams
### - Offensive and defensive stats
###### - Team, Year, Wins, Losses, R/G, PA, AB, R, H, 2B, 3B, HR, RBI, SB, BB, OBP, SLG, OPS, RA/G, ERA, H-allowed, SO-pitched, WHIP, etc.
### - Goal: Predict number of wins for a team in a season based on stats from above

Data Source: https://www.baseball-reference.com/

# Variance Inflation Factors (VIFs)

- Set 'wins' as my target. Compared all my potential features to analyze which statistics to include in my model.


<img src='figures/jnb-all-vifs.png'/>

# Eliminate Features

### Used cumulative stats rather than averages

###### 1B, 2B, 3B, HR, BB, SO, H-allowed, HR-allowed, BB-allowed, Errors

<!-- ![lessvifs](/Users/ryanholway/Documents/galvanize/capstone_I/figures/less-vifs.png) -->
<img src='figures/jnb-some-vifs.png'/>


<!-- ![Rsqd-pvals](/Users/ryanholway/Documents/galvanize/capstone_I/figures/rsqd-pvals.png) -->
<img src='figures/rsqd-pvals.png'/>

# EDA

### Offensive stats

<!-- ![hit-stats](/Users/ryanholway/Documents/galvanize/capstone_I/figures/hit-stats-fig.png) -->
<img src='figures/hit-stats-fig.png'/>


### Defensive / Pitching stats

<!-- ![pitch-stats](/Users/ryanholway/Documents/galvanize/capstone_I/figures/pitch-stats-fig.png) -->
<img src='figures/pitch-stats-fig.png'/>


### Misc.

<img src='figures/offensive-comparison.png'/>

<img src='figures/annual-trends.png'/>

# Lasso Regression

##### Training R-squared: 0.812
##### Training best alpha: 0.002
##### Test R-squared: 0.822

<!-- ![lasso-fig](/Users/ryanholway/Documents/galvanize/capstone_I/figures/lasso-fig.png) -->
<img src='figures/lasso-fig.png'/>

# Ridge Regression

##### Training R-squared: 0.812
##### Training best alpha: 8.286
##### Test R-squared: 0.821

<!-- ![ridge-fig](/Users/ryanholway/Documents/galvanize/capstone_I/figures/ridge-fig.png) -->
<img src='figures/ridge-fig.png'/>

# Predicted vs. Actual

#### Predicted target (Wins) from Ridge and Lasso models
- ridge.predict(X_std)
- lasso.predict(X_std)

#### Compared predicted targets from model to actual targets
- Ridge RMSLE: 0.063
- Lasso RMSLE: 0.063
- Ridge RMSE: 5.018
- Lasso RMSE: 5.014

## Residuals

##### Scatter plot of predicted wins and residuals

<!-- ![residuals](/Users/ryanholway/Documents/galvanize/capstone_I/figures/residuals.png) -->
<img src='figures/residuals.png'/>


#### QQ plots

<!-- ![QQplots](/Users/ryanholway/Documents/galvanize/capstone_I/figures/QQplots.png) -->
<img src='figures/QQplots.png'/>

# Fill it up again!

#### Re-ran Ridge and Lasso models
##### Trained on new features (OPS, ERA, WHIP, RBI, E)
##### Ridge
- Training R-squared: 0.873
- Test R-squared: 0.902
- RMSE: 4.013

##### Lasso
- Training R-squared: 0.873
- Test R-squared: 0.902
- RMSE: 4.010
