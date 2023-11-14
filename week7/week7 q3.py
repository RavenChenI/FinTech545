import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
F_F_3Factors = pd.read_csv('F-F_Research_Data_Factors_daily.csv', parse_dates=['Date'], index_col=0)
F_F_momentum = pd.read_csv('F-F_Momentum_Factor_daily.csv', parse_dates=['Date'], index_col=0)
data = F_F_3Factors.join(F_F_momentum, how = 'right')
data.dropna(inplace=True)
data/=100
data_10y = data.loc['2013-1-31':]
prices = pd.read_csv('DailyPrices.csv', parse_dates=['Date'], index_col=0)
returns = prices.pct_change()[1:]
stocks = ['AAPL', 'META', 'UNH', 'MA',
          'MSFT' ,'NVDA', 'HD', 'PFE',
          'AMZN' ,'BRK-B', 'PG', 'XOM',
          'TSLA' ,'JPM' ,'V', 'DIS',
          'GOOGL', 'JNJ', 'BAC', 'CSCO']
factors = ['Mkt-RF', 'SMB', 'HML', 'Mom   ']
stockset = returns[stocks]
dataset = stockset.join(data)
dataset.dropna(inplace = True)
dataset1 = dataset.loc["2013-01-31":"2023-01-31"]
avg_daily_return = pd.Series()
betas = pd.DataFrame(index=stocks, columns=factors)
alphas = pd.DataFrame(index=stocks, columns=['Alpha'])
for stock in stocks:
    y = dataset1[stock] - dataset1['RF']
    x = dataset1[factors]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    avg_daily_return[stock] = (model.params[factors]*data_10y[factors]).sum(axis = 1) + data_10y['RF'] + model.params['const']

annual_return = pd.Series()
for stock in stocks:
    annual_return[stock] = (1+avg_daily_return[stock]).cumprod()[-1]**(252/len(avg_daily_return[stock]))-1
geo_covariance = np.log(1 + returns[stocks]).cov() * 252

print(annual_return)


exp_returns = annual_return
risk_free_rate = 0.0425
cov_matrix = geo_covariance
# print(cov_matrix)
# Define a function that calculates the portfolio mean return and standard deviation
def portfolio_return_stddev(weights, exp_returns, cov_matrix):
    portfolio_return = np.sum(weights * exp_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_stddev

# Define a function that calculates the negative Sharpe ratio
def neg_sharpe_ratio(weights, exp_returns, cov_matrix, risk_free_rate):
    portfolio_return, portfolio_stddev = portfolio_return_stddev(weights, exp_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
    return -sharpe_ratio

num_stocks = len(exp_returns)
weights_0 = np.ones(num_stocks) / num_stocks

# Define the optimization constraints (i.e., weights must sum to 1)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Define the optimization bounds (i.e., weights must be between 0 and 1)
bounds = tuple((0, 1) for i in range(num_stocks))

# Use the SciPy minimize function to find the portfolio weights that maximize the Sharpe ratio
result = minimize(neg_sharpe_ratio, weights_0, args=(exp_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
weights_ef = result.x

# Define the additional constraints for the Super Effective Frontier
constraints_sef = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'eq', 'fun': lambda weights: portfolio_return_stddev(weights, exp_returns, cov_matrix)[0] - exp_returns[0]})

# Use the SciPy minimize function to find the portfolio weights that maximize the Sharpe ratio subject to the additional constraints
result_sef = minimize(neg_sharpe_ratio, weights_ef, args=(exp_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints_sef, bounds=bounds)
weights_sef = result_sef.x

weights_cml = pd.Series(index = annual_return.index)
i=0
for stock in stocks:
    weights_cml[stock] = round(weights_sef[i],3)*100
    i+=1

print(weights_cml)