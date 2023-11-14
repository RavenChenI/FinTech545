import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import inspect # delta normal

portfolios = pd.read_csv('problem2.csv')
portfolios['CurrentValue'] = portfolios['CurrentPrice'] * portfolios['Holding']

S = 165
N = 20
current_date = pd.to_datetime('2023-03-03')
div_date = pd.to_datetime('2023-03-15')
r = 0.0425
div = 1


def calculate_var(data, mean=0, alpha=0.05):
    return mean - np.quantile(data, alpha)


def calculate_es(data, mean=0, alpha=0.05):
    return -np.mean(data[data < -calculate_var(data, mean, alpha)])


def bt_american_without_div(call, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call == 'call' else -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nNodes = nNodeFunc(N)
    optionValues = [0.0] * nNodes

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * u ** i * d ** (j - i)
            optionValues[idx] = max(0, z * (price - strike))
            if j < N:
                optionValues[idx] = max(optionValues[idx], df * (
                        pu * optionValues[idxFunc(i + 1, j + 1)] + pd * optionValues[idxFunc(i, j + 1)]))

    return optionValues[0]


def bt_american(call, underlying, strike, ttm, rf, divAmts, divTimes, ivol, N):
    # if there are no dividends or the first dividend is outside out grid, return the standard bt_american value
    if not divAmts or not divTimes or divTimes[0] > N:
        return bt_american_without_div(call, underlying, strike, ttm, rf, rf, ivol, N)

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call == 'call' else -1

    def n_node_func(n):
        return int((n + 1) * (n + 2) / 2)

    def idx_func(i, j):
        return n_node_func(j - 1) + i

    n_nodes = n_node_func(divTimes[0])

    option_values = np.empty(n_nodes)

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idx_func(i, j)
            price = underlying * u ** i * d ** (j - i)

            if j < divTimes[0]:
                # times before the dividend working backward induction
                option_values[idx] = max(0, z * (price - strike))
                option_values[idx] = max(option_values[idx], df * (
                            pu * option_values[idx_func(i + 1, j + 1)] + pd * option_values[idx_func(i, j + 1)]))
            else:
                # time of the dividend
                val_no_exercise = bt_american(call, price - divAmts[0], strike, ttm - divTimes[0] * dt, rf, divAmts[1:],
                                              [t - divTimes[0] for t in divTimes[1:]], ivol, N - divTimes[0])
                val_exercise = max(0, z * (price - strike))
                option_values[idx] = max(val_no_exercise, val_exercise)

    return option_values[0]


def implied_vol_american(call, underlying, strike, ttm, rf, divAmts, divTimes, N, market_price, x0=0.5):
    def equation(ivol):
        return bt_american(call, underlying, strike, ttm, rf, divAmts, divTimes, ivol, N) - market_price

    # Back solve the binomial tree valuation to get the implied volatility
    return fsolve(equation, x0=x0, xtol=0.0001)[0]


implied_vols = []
for i in range(len(portfolios.index)):
    if portfolios["Type"][i] == "Stock":
        implied_vols.append(None)
    else:
        option_type = portfolios["OptionType"][i].lower()
        X = portfolios["Strike"][i]
        T = (pd.to_datetime(portfolios["ExpirationDate"][i]) - current_date).days / 365
        div_time = int(
            (div_date - current_date).days / (pd.to_datetime(portfolios["ExpirationDate"][i]) - current_date).days * N)
        market_price = portfolios["CurrentPrice"][i]
        sigma = implied_vol_american(option_type, S, X, T, r, [div], [div_time], N, market_price)
        implied_vols.append(sigma)

# Store the implied volatility in portfolios
portfolios["ImpliedVol"] = implied_vols


def sim_result(portfolio, sim_prices):
    days_ahead = 10
    sim_value = pd.DataFrame(index=portfolio.index, columns=list(range(len(sim_prices))))
    for i in portfolio.index:
        if portfolio['Type'][i] == 'Stock':
            option_values = sim_prices
        else:
            option_type = portfolio['OptionType'][i].lower()
            X = float(portfolio['Strike'][i])
            T = ((pd.to_datetime(portfolio['ExpirationDate'][i]) - current_date).days - days_ahead) / 365
            r = 0.0425
            ivol = float(portfolio['ImpliedVol'][i])
            divTimes = int(((div_date - current_date).days - days_ahead) / (
                        pd.to_datetime(portfolios["ExpirationDate"][i]) - current_date).days * N)
            divAmts = 1
            option_values = []
            for S in sim_prices:
                option_values.append(bt_american(option_type, S, X, T, r, [divAmts], [divTimes], ivol, N))
        sim_value.loc[i, :] = portfolio["Holding"][i] * np.array(option_values)
    sim_value['Portfolio'] = portfolio['Portfolio']
    return sim_value.groupby('Portfolio').sum()


sim_prices = np.linspace(120, 220, 20)
all_prices = pd.read_csv("DailyPrices.csv")

aapl_prices = all_prices['AAPL']
aapl_returns = aapl_prices.pct_change()[1:] - np.mean(aapl_prices.pct_change()[1:])
# Simulate the prices based on returns with normal distribution
std = aapl_returns.std()
sim_returns = norm(0, std).rvs((10, 100))
sim_prices = S * (1 + sim_returns).prod(axis=0)

curr_values = portfolios.groupby('Portfolio')['CurrentValue'].sum()
sim_values = sim_result(portfolios, sim_prices)

# Calculate the value difference
sim_value_changes = (sim_values.T - curr_values).T

# Calculate the Mean, VaR and ES, and print the results
result = pd.DataFrame(index=sim_value_changes.index)
result['Mean'] = sim_value_changes.mean(axis=1)
result['VaR'] = sim_value_changes.apply(lambda x: calculate_var(x, 0), axis=1)
result['ES'] = sim_value_changes.apply(lambda x: calculate_es(x, 0), axis=1)
print(result)


# calculate first order derivative
def first_order_der(func, x, delta):
    return (func(x + delta) - func(x - delta)) / (2 * delta)


# calculate second order derivative
def second_order_der(func, x, delta):
    return (func(x + delta) + func(x - delta) - 2 * func(x)) / delta ** 2


def cal_partial_derivative(func, order, arg_name, delta=1e-3):
    # initialize for argument names and order
    arg_names = list(inspect.signature(func).parameters.keys())
    derivative_fs = {1: first_order_der, 2: second_order_der}

    def partial_derivative(*args, **kwargs):
        # parse argument names and order
        args_dict = dict(list(zip(arg_names, args)) + list(kwargs.items()))
        arg_val = args_dict.pop(arg_name)

        def partial_f(x):
            p_kwargs = {arg_name: x, **args_dict}
            return func(**p_kwargs)

        return derivative_fs[order](partial_f, arg_val, delta)

    return partial_derivative


S = 165
N = 20
current_date = pd.to_datetime('2023-03-03')
div_date = pd.to_datetime('2023-03-15')
r = 0.0425
div = 1

cal_amr_delta_num = cal_partial_derivative(bt_american, 1, 'underlying')

# Calculate the implied volatility for all portfolios
deltas = []
for i in range(len(portfolios.index)):
    if portfolios["Type"][i] == "Stock":
        deltas.append(1)
    else:
        option_type = portfolios["OptionType"][i].lower()
        X = portfolios["Strike"][i]
        T = ((pd.to_datetime(portfolios["ExpirationDate"][i]) - current_date).days - 10) / 365
        div_time = int(
            (div_date - current_date).days / (pd.to_datetime(portfolios["ExpirationDate"][i]) - current_date).days * N)
        delta = cal_amr_delta_num(option_type, S, X, T, r, [div], [div_time], sigma, N)
        deltas.append(delta)

# Store the deltas in portfolios
portfolios["deltas"] = deltas
alpha = 0.05
t = 10
result_dn = pd.DataFrame(index=sorted(portfolios['Portfolio'].unique()), columns=['Mean', 'VaR', 'ES'])
result_dn.index.name = 'Portfolio'
for pfl, df in portfolios.groupby('Portfolio'):
    gradient = S / df['CurrentValue'].sum() * (df['Holding'] * df['deltas']).sum()
    pfl_10d_std = abs(gradient) * std * np.sqrt(t)
    N = norm(0, 1)
    present_value = df['CurrentValue'].sum()
    result_dn.loc[pfl]['Mean'] = 0
    result_dn.loc[pfl]['VaR'] = -present_value * N.ppf(alpha) * pfl_10d_std
    result_dn.loc[pfl]['ES'] = present_value * pfl_10d_std * N.pdf(N.ppf(alpha)) / alpha

print(result_dn)
