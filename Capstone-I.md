# Capstone I

# Analyzing MLB statistics to predict wins

### - 30 teams in MLB since 1998
### - took team statistics from 1998 until 2018
### - total of 630 teams
### - offensive and defensive stats
###### - Team, Year, Wins, Losses, R/G, PA, AB, R, H, 2B, 3B, HR, RBI, SB, BB, OBP, SLG, OPS, RA/G, ERA, H-allowed, SO-pitched, WHIP, etc.

# Variance Inflation Factors (VIFs)

![VIFfactor](/Users/ryanholway/Documents/galvanize/capstone_I/figures/high-vifs.png)

# Eliminate Features

![VIFfactor](/Users/ryanholway/Documents/galvanize/capstone_I/figures/less-vifs.png)

![Rsqd-pvals](/Users/ryanholway/Documents/galvanize/capstone_I/figures/rsqd-pvals.png)

# EDA

### Offensive stats

![hit-stats](/Users/ryanholway/Documents/galvanize/capstone_I/figures/hit-stats-fig.png)


### Defensive stats

![pitch-stats](/Users/ryanholway/Documents/galvanize/capstone_I/figures/pitch-stats-fig.png)


# Lasso Regression

##### Training R-squared: 0.812
##### Training best alpha: 0.002
##### Test R-squared: 0.822

![pitch-stats](/Users/ryanholway/Documents/galvanize/capstone_I/figures/lasso-fig.png)


# Ridge Regression

##### Training R-squared: 0.812
##### Training best alpha: 8.286
##### Test R-squared: 0.821

![pitch-stats](/Users/ryanholway/Documents/galvanize/capstone_I/figures/ridge-fig.png)
