import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t

# problem 1
# assume r is between -1 ~ 1 -> 3σ = 1 -> σ = 0.3
price = 1
mean = 0
sigma = 0.3
r = np.random.normal(mean, sigma, 10000)
# Classical Brownian Motion
price_t = price + r
print('Classical Brownian Motion')
print(np.mean(price_t))
print(np.std(price_t))

# Arithmetic Return System
price_t = price * (1 + r)
print('Arithmetic Return System')
print(np.mean(price_t))
print(np.std(price_t))

# Log Return or Geometric Brownian Motion
price_t = price * np.exp(r)
print('Log Return or Geometric Brownian Motion')
print(np.mean(price_t))
print(np.std(price_t))

# problem 2
data = pd.read_csv('DailyPrices.csv')


# print(data)
def ror(price):
    ror = []
    for i in range(len(price) - 1):
        ror.append(price[i + 1] / price[i])
    ror = np.array(ror)
    return ror - 1


data = data.drop(['Date'], axis=1)

ror_table = pd.DataFrame()
for column in data.columns:
    returns = ror(data[column])
    ror_table[column] = returns

print(ror_table)

# Remove the mean from the series so that the mean(META)=0
meta = ror_table["META"]
meta = meta - np.mean(meta)


# Calculate VAR using a normal distribution
def calculate_var(data, alpha=0.05):
    return -np.quantile(data, alpha)


sigma = np.sqrt(np.var(meta))
norm = np.random.normal(0, sigma, 1000)
var_norm = calculate_var(norm, alpha=0.05)
print(round(var_norm * 100, 2), '%')


# Exponentially Weighted variance
def weight_gen(n, lambd):
    weight = np.zeros(n)
    for i in range(n):
        weight[i] = (1 - lambd) * (lambd) ** i
    normalized_weight = weight / np.sum(weight)
    return normalized_weight[::-1]


def cov_gen(data, weight):
    data = data - data.mean()
    weight = np.diag(weight)
    data_left = weight @ data
    data_right = np.dot(data.T, data_left)
    return data_right


# Calculate VAR using a normal distribution with an Exponentially Weighted variance lambda = 0.94
weight = weight_gen(len(meta), 0.94)
cov = cov_gen(np.matrix(meta).T, weight)
sigma = np.sqrt(cov[0, 0])
norm = np.random.normal(0, sigma, 1000)
var_ew_norm = calculate_var(norm, alpha=0.05)
print(var_ew_norm * 100, '%')

# Calculate VAR using an MLE fitted T distribution

result = t.fit(meta, method="MLE")
df = result[0]
loc = result[1]
scale = result[2]
simulation_t = t(df, loc, scale).rvs(10000)
var_t = calculate_var(simulation_t)
print(var_t * 100, '%')

# Calculate VAR using a fitted AR(1) model

# fitting AR(1) model
model = ARIMA(meta, order=(1, 0, 0))
res = model.fit()
# print(res.summary())
sigma = np.sqrt(res.params['sigma2'])
norm = np.random.normal(0, sigma, 1000)
ar1_norm = calculate_var(norm, alpha=0.05)
print(ar1_norm * 100, '%')

# Calculate VAR using a Historic Simulation
var_historic = calculate_var(meta)
print(var_historic * 100, '%')



#Problem 3
import numpy as np
import pandas as pd

portfolio = pd.read_csv('portfolio.csv')
data = pd.read_csv('DailyPrices.csv')
portfolioA = portfolio[portfolio['Portfolio'] == 'A']
portfolioB = portfolio[portfolio['Portfolio'] == 'B']
portfolioC = portfolio[portfolio['Portfolio'] == 'C']


def portfolio_pv(data, portfolio):
    pv = []
    for stock in portfolio['Stock']:
        pv.append(data.iloc[-1][stock])
    return np.array(pv)
def weight(data, portfolio):
    total = np.sum(portfolio['Holding']*100*portfolio_pv(data, portfolio))
    weight =[]
    for stock in portfolio['Stock']:
        weight.append(portfolio[portfolio['Stock'] == stock]['Holding']*100*data.iloc[-1][stock]/total)
    return weight
def calculate_return(price, method = 'discrete'):
    if method == 'discrete':
        return price.pct_change()
    if method == 'log':
        return np.log(price / price.shift(1))
def cov_ew(data, portfolio, method = 'discrete'):
    price = data[portfolio['Stock'].values]
    returns = calculate_return(price, method = method)
    weight = weight_gen(len(returns.dropna().iloc[0:]), 0.94)
    cov = cov_gen(returns.dropna(), weight)
    return cov



portfolio = [portfolioA, portfolioB, portfolioC]
sumtotal = 0
z_score = 1.645
for portfolio in portfolio:
    holding = portfolio['Holding']*100
    pv = portfolio_pv(data, portfolio)
    cov_matrix = cov_ew(data, portfolio, method = 'discrete')
    weights = np.array(weight(data, portfolio))
    middle = np.dot(cov_matrix, weights)
    var = np.dot(weights.T, middle)
    present = np.sum(pv * holding)
    total = present * z_score * np.sqrt(var)
    total = round(total[0][0],2)
    sumtotal+=total
    print('VaR for portfolio', portfolio['Portfolio'].iloc[0], 'is $', total)
print('VaR for total portfolio is $', sumtotal)


portfolio = [portfolioA, portfolioB, portfolioC]
sumtotal = 0
for portfolio in portfolio:
    holding = portfolio['Holding']*100
    pv = portfolio_pv(data, portfolio)
    price = data[portfolio['Stock'].values]
    returns = calculate_return(price, method = 'log')
    cov_matrix = returns.cov()
    weights = np.array(weight(data, portfolio))
    middle = np.dot(cov_matrix, weights)
    var = np.dot(weights.T, middle)
    present = np.sum(pv * holding)
    total = present * np.sqrt(var)
    total = round(total[0][0],2)
    sumtotal+=total
    print('VaR for portfolio', portfolio['Portfolio'].iloc[0], 'is $', total)
print('VaR for total portfolio is $', round(sumtotal,2))