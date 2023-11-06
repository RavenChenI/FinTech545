import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve
from statsmodels.tsa.arima.model import ARIMA

# Problem 1

S = 165
start = pd.to_datetime('2023-03-03')
expiration = pd.to_datetime('2023-03-17')
rf = 0.0525
i = 0.0053
b = rf - i
ttm = (expiration - start).days
T = ttm / 365  # Calculate the time to maturity using calendar days (not trading days)
print("Time to Maturity is", ttm, "days")
sigma = np.linspace(0.1, 0.8, 50)
#
#
def option_value(S, X, rf, b, sigma, T):
    d1 = (np.log(S / X) + (b + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Call = S * np.exp((b - rf) * T) * norm.cdf(d1) - X * np.exp(-rf * T) * norm.cdf(d2)
    Put = X * np.exp(-rf * T) * norm.cdf(-d2) - S * np.exp((b - rf) * T) * norm.cdf(-d1)
    plt.plot(sigma, Call, label="Call")
    plt.plot(sigma, Put, label="Put")
    plt.xlabel('Implied Volatility')
    plt.ylabel('Option Value')
    plt.legend()
    plt.show()


X = 165
print("At the Money")
option_value(S, X, rf, b, sigma, T)
X = 200  # Call -> Out of Money; Put -> In the Money
print("X = 200 ; Call -> Out of Money; Put -> In the Money")
option_value(S, X, rf, b, sigma, T)
X = 100
print("X = 100 ; Call -> In the Money; Put -> Out of the Money")
option_value(S, X, rf, b, sigma, T)

# Problem 2
df = pd.read_csv("AAPL_Options.csv")
start = pd.to_datetime('2023-10-30')
S = 170.15
rf = 0.0525
i = 0.0057
b = rf - i
df['T'] = (pd.to_datetime(df['Expiration']) - start).dt.days / 365


def call_or_put(option_type, S, X, rf, b, sigma, T):
    d1 = (np.log(S / X) + (b + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    print(sigma)
    print(type(sigma))
    print(norm.cdf(d2))
    print(type(norm.cdf(d2)))
    if option_type == "call":
        Call = S * np.exp((b - rf) * T) * norm.cdf(d1) - X * np.exp(-rf * T) * norm.cdf(d2)
        return Call
    if option_type == "put":
        Put = X * np.exp(-rf * T) * norm.cdf(-d2) - S * np.exp((b - rf) * T) * norm.cdf(-d1)
        return Put



def implied_vol(option_type, S, X, T, r, b, market_price, x0=0.2):
    def equation(sigma):
        return call_or_put(option_type, S, X, r, b, sigma, T) - market_price

    # Back solve the Black-Scholes formula to get the implied volatility
    return fsolve(equation, x0=x0, xtol=0.0001)[0]


call_market = df[df["Type"] == "Call"]["Last Price"].tolist()
put_market = df[df["Type"] == "Put"]["Last Price"].tolist()
call_strike = df[df["Type"] == "Call"]["Strike"].tolist()
put_strike = df[df["Type"] == "Put"]["Strike"].tolist()
call_T = df[df["Type"] == "Call"]["T"].tolist()
put_T = df[df["Type"] == "Put"]["T"].tolist()
call_iv = []
put_iv = []
for i in range(len(call_market)):
    call_iv.append(implied_vol('call', S, call_strike[i], call_T[i], rf, b, call_market[i]))
    put_iv.append(implied_vol('put', S, put_strike[i], put_T[i], rf, b, put_market[i]))

plt.plot(call_strike, call_iv, label='Call')
plt.plot(put_strike, put_iv, label='Put')
plt.axvline(x=S, color='r', linestyle='--')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.legend()
plt.show()

# Problem 3

start = pd.to_datetime('2023-10-30')
S = 170.15
rf = 0.0525
i = 0.0057
b = rf - i
portfolio = pd.read_csv('problem3.csv')
price = np.linspace(100, 200, 100)


def payoff_call(S, X):
    return np.maximum(S - X, 0)


def payoff_put(S, X):
    return np.maximum(X - S, 0)


expiration = pd.to_datetime('2023-12-15')
T = (expiration - start).days / 365

# 1. Straddle
df1 = portfolio[portfolio['Portfolio'] == 'Straddle']
payoff = payoff_call(price, df1['Strike'].values[0]) + df1[df1['OptionType'] == 'Put']['Holding'].values[
    0] * payoff_put(price, df1['Strike'].values[0]) - df1['CurrentPrice'].values[1] - df1['CurrentPrice'].values[0]
plt.plot(price, payoff)
plt.title('Straddle')
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()

# 2. SynLong
df1 = portfolio[portfolio['Portfolio'] == 'SynLong']
payoff = df1[df1['OptionType'] == 'Call']['Holding'].values[0] * payoff_call(price, df1['Strike'].values[0]) + \
         df1[df1['OptionType'] == 'Put']['Holding'].values[0] * payoff_put(price, df1['Strike'].values[0]) + \
         df1['CurrentPrice'].values[1] - df1['CurrentPrice'].values[0]
plt.plot(price, payoff)
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('SynLong')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()

# 3. CallSpread
df1 = portfolio[portfolio['Portfolio'] == 'CallSpread']
payoff = df1[df1['OptionType'] == 'Call']['Holding'].values[0] * payoff_call(price, df1['Strike'].values[0]) + \
         df1[df1['OptionType'] == 'Call']['Holding'].values[1] * payoff_call(price, df1['Strike'].values[1]) + \
         df1['CurrentPrice'].values[1] - df1['CurrentPrice'].values[0]
plt.plot(price, payoff)
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Call Spread')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()

# 4. PutSpread
df1 = portfolio[portfolio['Portfolio'] == 'PutSpread']
payoff = df1[df1['OptionType'] == 'Put']['Holding'].values[1] * payoff_put(price, df1['Strike'].values[1]) + \
         df1[df1['OptionType'] == 'Put']['Holding'].values[0] * payoff_put(price, df1['Strike'].values[0]) + \
         df1['CurrentPrice'].values[1] - df1['CurrentPrice'].values[0]
plt.plot(price, payoff)
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('PutSpread')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()

# 5. Stock
df1 = portfolio[portfolio['Portfolio'] == 'Stock']
payoff = df1['Holding'].values[0] * price - S
plt.plot(price, payoff)
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Stock')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()

# 6. Call
df1 = portfolio[portfolio['Portfolio'] == 'Call ']
payoff = payoff_call(price, df1['Strike'].values[0]) - df1['CurrentPrice'].values[0]
plt.plot(price, payoff)
plt.title('Call')
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()

# 7. Put
df1 = portfolio[portfolio['Portfolio'] == 'Put ']
payoff = payoff_put(price, df1['Strike'].values[0]) - df1['CurrentPrice'].values[0]
plt.plot(price, payoff)
plt.title('Put')
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()

# 8. CoveredCall
df1 = portfolio[portfolio['Portfolio'] == 'CoveredCall']
payoff = df1['CurrentPrice'].values[1] - np.maximum(price - df1['Strike'].values[1], 0) + price - S
plt.plot(price, payoff)
plt.title('CoveredCall')
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()

# 9. ProtectedPut
df1 = portfolio[portfolio['Portfolio'] == 'ProtectedPut']
payoff = df1[df1['OptionType'] == 'Put']['Holding'].values[0] * payoff_put(price, df1['Strike'].values[1]) - \
         df1['CurrentPrice'].values[1] + df1['Holding'].values[0] * price - S
plt.plot(price, payoff)
plt.title('ProtectedPut')
plt.axvline(x=S, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Underlying Price')
plt.ylabel('Portfolio Value')
plt.show()


# ---------------------------
# Using DailyPrices.csv. Calculate the log returns of AAPL. Demean the series so there is 0 mean. Fit an
# AR(1) model to AAPL returns. Simulate AAPL returns 10 days ahead and apply those returns to the
# current AAPL price (above). Calculate Mean, VaR and ES. Discuss.
def return_calculate(price, method='discrete'):
    returns = []
    for i in range(len(price) - 1):
        returns.append(price[i + 1] / price[i])
    returns = np.array(returns)
    if method == 'discrete':
        return returns - 1
    if method == 'log':
        return np.log(returns)


dp = pd.read_csv('DailyPrices.csv', index_col=0)
aapl = dp['AAPL']
aapl_return = return_calculate(aapl, 'log')
aapl_return = aapl_return - aapl_return.mean()
# fitting AR(1) model
model = ARIMA(aapl_return, order=(1, 0, 0))
res = model.fit()
print(res.summary())

#Simulating 10 days ahead, with 10000 simulations
sigma = np.sqrt(res.params[2])
norm_random = np.random.normal(0, sigma, size = (10,10000))
# sim_prices is the 10th date price
sim_prices = S * (1+norm_random).prod(axis=0)


# Calculating current portfolio value respectively
portfolio['CurrentValue'] = portfolio['Holding']*portfolio['CurrentPrice']
current_value = portfolio.groupby('Portfolio')['CurrentValue'].sum()

iv =[]
for i in portfolio.index:
    option_type = portfolio['OptionType'][i]
    if isinstance(option_type, float):
        iv.append('NaN')
    else:
        S = 170.15
        X = float(portfolio['Strike'][i])
        T = (pd.to_datetime(portfolio['ExpirationDate'][i]) - pd.to_datetime('2023-10-30')).days/365
        r = 0.0525
        q = 0.0057
        b = r-q
        market_price = float(portfolio['CurrentPrice'][i])
        iv.append(implied_vol(option_type.lower(), S, X, T, r, b, market_price, x0=0.5))
portfolio['IV'] = iv

def sim_result(portfolio, sim_prices):
    days_ahead = 10
    sim_value = pd.DataFrame(index = portfolio.index, columns = list(range(len(sim_prices))))
    for i in portfolio.index:
        if portfolio['Type'][i] == 'Stock':
            final_value = sim_prices
        else:
            option_type = portfolio['OptionType'][i].lower()
            S = sim_prices
            X = float(portfolio['Strike'][i])
            T = ((pd.to_datetime(portfolio['ExpirationDate'][i]) - pd.to_datetime('2023-10-30')).days - days_ahead)/365
            r = 0.0525
            q = 0.0057
            b = r-q
            sigma = float(portfolio['IV'][i])
            final_value = call_or_put(option_type, S, X, r, b, sigma, T)
        sim_value.loc[i, :] = portfolio["Holding"][i] * final_value
    sim_value['Portfolio'] = portfolio['Portfolio']
    return sim_value.groupby('Portfolio').sum()

def calculate_var(data, mean=0, alpha=0.05):
    return mean-np.quantile(data, alpha)

def calculate_es(data, mean = 0, alpha=0.05):
    return -np.mean(data[data<-calculate_var(data, mean, alpha)])




sim_results = sim_result(portfolio, sim_prices)
sim_result_difference = sim_results.sub(current_value, axis=0).T
result = pd.DataFrame(index=sim_results.index)
result['Mean'] = sim_result_difference.mean(axis=0)
result['VaR'] = sim_result_difference.apply(lambda x:calculate_var(x,0), axis=0)
result['ES'] = sim_result_difference.apply(lambda x:calculate_es(x,0), axis=0)
print(result)
