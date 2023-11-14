import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

# Problem 1
S = 151.03
X = 165
c_date = pd.to_datetime('2023-03-13')
option_exp = pd.to_datetime('2023-04-15')
r = 0.0425
q = 0.0053
sigma = 0.2
b = r-q
T = (option_exp - c_date).days/365
def d1(S, K, b, sigma, T):
    return (np.log(S/K) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))

def d2(S, K, b, sigma, T):
    return d1(S, K, b, sigma, T) - sigma*np.sqrt(T)

def gbsm(option_type, S, X, r, b, sigma, T):
    d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S*np.exp((b-r)*T)*norm.cdf(d1) - X*np.exp(-r*T)*norm.cdf(d2)
    else:
        return X*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp((b-r)*T)*norm.cdf(-d1)


def implied_vol(option_type, S, X, T, r, b, market_price, x0=0.5):
    def equation(sigma):
        return gbsm(option_type, S, X, r, b, sigma, T) - market_price
    # Back solve the Black-Scholes formula to get the implied volatility
    return fsolve(equation, x0=x0, xtol=0.0001)[0]

def closed_form_greeks(S, X, b, sigma, T):
    delta_call = np.exp((b-r)*T)*norm.cdf(d1(S, X, b, sigma, T))
    print('Delta of the call option is: ', round(delta_call,3))
    delta_put = np.exp((b-r)*T)*(norm.cdf(d1(S, X, b, sigma, T)) - 1)
    print('Delta of the put option is: ', round(delta_put,3))
    gamma = np.exp((b-r)*T)*norm.pdf(d1(S, X, b, sigma, T))/(S*sigma*np.sqrt(T))
    print('Gamma of the call option is: ', round(gamma,3))
    print('Gamma of the put option is: ', round(gamma,3))
    vega = S*np.exp((b-r)*T)*norm.pdf(d1(S, X, b, sigma, T))*np.sqrt(T)
    print('Vega of the call option is: ', round(vega,3))
    print('Vega of the put option is: ', round(vega,3))
    theta_call = -S*np.exp((b-r)*T)*norm.pdf(d1(S, X, b, sigma, T))*sigma/(2*np.sqrt(T)) - (b-r)*S*np.exp((b-r)*T)*norm.cdf(d1(S, X, b, sigma, T)) - r*X*np.exp(-r*T)*norm.cdf(d2(S, X, b, sigma, T))
    print('Theta of the call option is: ', round(theta_call,3))
    theta_put = -S*np.exp((b-r)*T)*norm.pdf(d1(S, X, b, sigma, T))*sigma/(2*np.sqrt(T)) + (b-r)*S*np.exp((b-r)*T)*norm.cdf(-d1(S, X, b, sigma, T)) + r*X*np.exp(-r*T)*norm.cdf(-d2(S, X, b, sigma, T))
    print('Theta of the put option is: ', round(theta_put,3))
    # because the textbook has an assumption that b = rf but it does not hold here, we need to calculate the rho seperately
    rho_call = -T*S*np.exp(b*T - r*T)*norm.cdf(d1(S, X, b, sigma, T)) + X*T*np.exp(-r*T)*norm.cdf(d2(S, X, b, sigma, T))
    print('Rho of the call option is: ', round(rho_call,3))
    rho_put = -X*T*np.exp(-r*T)*norm.cdf(-d2(S, X, b, sigma, T))+T*S*np.exp(b*T - r*T)*norm.cdf(-d1(S, X, b, sigma, T))
    print('Rho of the put option is: ', round(rho_put,3))
    carry_rho_call = S*T*np.exp((b-r)*T)*norm.cdf(d1(S, X, b, sigma, T))
    print('Carry Rho of the call option is: ', round(carry_rho_call,3))
    carry_rho_put = -S*T*np.exp((b-r)*T)*norm.cdf(-d1(S, X, b, sigma, T))
    print('Carry Rho of the put option is: ', round(carry_rho_put,3))

print('closed_form_greeks')
closed_form_greeks(S, X, b, sigma, T)

def finite_diff_greeks(S, X, r, b, sigma, T):
    delta_call_fd = (gbsm('call', S, X, r, b, sigma, T) - gbsm('call', S-0.001, X, r, b, sigma, T))/0.001
    print('Delta of the call option is: ', round(delta_call_fd,3))
    delta_put_fd = (gbsm('put', S, X, r, b, sigma, T) - gbsm('put', S-0.001, X, r, b, sigma, T))/0.001
    print('Delta of the put option is: ', round(delta_put_fd,3))
    gamma_call_fd = (np.exp((b-r)*T)*norm.cdf(d1(S, X, b, sigma, T)) - np.exp((b-r)*T)*norm.cdf(d1(S-0.001, X, b, sigma, T)))/0.001
    print('Gamma of the call option is: ', round(gamma_call_fd,3))
    gamma_put_fd = (np.exp((b-r)*T)*(norm.cdf(d1(S, X, b, sigma, T)) - 1) - np.exp((b-r)*T)*(norm.cdf(d1(S-0.001, X, b, sigma, T)) - 1))/0.001
    print('Gamma of the put option is: ', round(gamma_put_fd,3))
    vega_call_fd = (gbsm('call', S, X, r, b, sigma, T) - gbsm('call', S, X, r, b, sigma-0.001, T))/0.001
    print('Vega of the call option is: ', round(vega_call_fd,3))
    vega_put_fd = (gbsm('put', S, X, r, b, sigma, T) - gbsm('put', S, X, r, b, sigma-0.001, T))/0.001
    print('Vega of the put option is: ', round(vega_put_fd,3))
    theta_call_fd = -(gbsm('call', S, X, r, b, sigma, T) - gbsm('call', S, X, r, b, sigma, T-0.001))/0.001
    print('Theta of the call option is: ', round(theta_call_fd,3))
    theta_put_fd = -(gbsm('put', S, X, r, b, sigma, T) - gbsm('put', S, X, r, b, sigma, T-0.001))/0.001
    print('Theta of the put option is: ', round(theta_put_fd,3))
    rho_call_fd = (gbsm('call', S, X, r, b, sigma, T) - gbsm('call', S, X, r-0.001, b, sigma, T))/0.001
    print('Rho of the call option is: ', round(rho_call_fd,3))
    rho_put_fd = (gbsm('put', S, X, r, b, sigma, T) - gbsm('put', S, X, r-0.001, b, sigma, T))/0.001
    print('Rho of the put option is: ', round(rho_put_fd,3))
    carry_rho_call_fd = (gbsm('call', S, X, r, b, sigma, T) - gbsm('call', S, X, r, b-0.001, sigma, T))/0.001
    print('Carry Rho of the call option is: ', round(carry_rho_call_fd,3))
    carry_rho_put_fd = (gbsm('put', S, X, r, b, sigma, T) - gbsm('put', S, X, r, b-0.001, sigma, T))/0.001
    print('Carry Rho of the put option is: ', round(carry_rho_put_fd,3))
print("")
print('finite_diff_greeks')
finite_diff_greeks(S, X, r, b, sigma, T)

# Implement the binomial tree valuation for American options without dividends
S = 151.03
X = 165
c_date = pd.to_datetime('2022-03-13')
option_exp = pd.to_datetime('2022-04-15')
r = 0.0425
q = 0.0053
sigma = 0.2
b = r - q
T = (option_exp - c_date).days / 365
N = 200  # number of steps in tree, suppose 200


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

print("")
call_without_div = bt_american_without_div('call', S, X, T, r, b, sigma, N)
print('The american call option value without dividend is: ', round(call_without_div, 3))
put_without_div = bt_american_without_div('put', S, X, T, r, b, sigma, N)
print('The american put option value without dividend is: ', round(put_without_div, 3))

closed_form_greeks(S, X, b, sigma, T)

# Implement the binomial tree valuation for American options with dividends
S = 151.03
X = 165
c_date = pd.to_datetime('2022-03-13')
option_exp = pd.to_datetime('2022-04-15')
r = 0.0425
q = 0.0053
sigma = 0.2
T = (option_exp - c_date).days/365
dividen = 0.88
dividen_date = pd.to_datetime('2022-04-11')
div_time = int((dividen_date - c_date).days/(option_exp - c_date).days * N)
N = 200


def bt_american(call, underlying, strike, ttm, rf, divAmts, divTimes, ivol, N):

    #if there are no dividends or the first dividend is outside out grid, return the standard bt_american value
    if not divAmts or not divTimes or divTimes[0] > N:
        return bt_american_without_div(call, underlying, strike, ttm, rf, rf, ivol, N)

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call=='call' else -1

    def n_node_func(n):
        return int((n + 1) * (n + 2) / 2)

    def idx_func(i, j):
        return n_node_func(j - 1) + i

    n_nodes = n_node_func(divTimes[0])

    option_values = np.empty(n_nodes)

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idx_func(i, j)
            price = underlying * u**i * d**(j-i)

            if j < divTimes[0]:
                # times before the dividend working backward induction
                option_values[idx] = max(0, z * (price - strike))
                option_values[idx] = max(option_values[idx], df * (pu * option_values[idx_func(i+1, j+1)] + pd * option_values[idx_func(i, j+1)]))
            else:
                # time of the dividend
                val_no_exercise = bt_american(call, price - divAmts[0], strike, ttm - divTimes[0] * dt, rf, divAmts[1:], [t - divTimes[0] for t in divTimes[1:]], ivol, N - divTimes[0])
                val_exercise = max(0, z * (price - strike))
                option_values[idx] = max(val_no_exercise, val_exercise)

    return option_values[0]
print("")
print('The american call option value with dividend is: ', round(bt_american('call', S, X, T, r, [dividen], [div_time], sigma, N),3))
print('The american put option value with dividend is: ', round(bt_american('put', S, X, T, r, [dividen], [div_time], sigma, N),3))
delta_call_fd = (bt_american('call', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('call', S-0.001, X, T, r, [dividen], [div_time], sigma, N))/0.001
print('Delta of the call option is: ', round(delta_call_fd,3))
delta_put_fd = (bt_american('put', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('put', S-0.001, X, T, r, [dividen], [div_time], sigma, N))/0.001
print('Delta of the put option is: ', round(delta_put_fd,3))
gamma_call_fd = (np.exp((b-r)*T)*norm.cdf(d1(S, X, b, sigma, T)) - np.exp((b-r)*T)*norm.cdf(d1(S-0.001, X, b, sigma, T)))/0.001
print('Gamma of the call option is: ', round(gamma_call_fd,3))
gamma_put_fd = (np.exp((b-r)*T)*(norm.cdf(d1(S, X, b, sigma, T)) - 1) - np.exp((b-r)*T)*(norm.cdf(d1(S-0.001, X, b, sigma, T)) - 1))/0.001
print('Gamma of the put option is: ', round(gamma_put_fd,3))
vega_call_fd = (bt_american('call', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('call', S, X, T, r, [dividen], [div_time], sigma-0.001, N))/0.001
print('Vega of the call option is: ', round(vega_call_fd,3))
vega_put_fd = (bt_american('put', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('put', S, X, T, r, [dividen], [div_time], sigma-0.001, N))/0.001
print('Vega of the put option is: ', round(vega_put_fd,3))
theta_call_fd = -(bt_american('call', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('call', S, X, T-0.001, r, [dividen], [div_time], sigma, N))/0.001
print('Theta of the call option is: ', round(theta_call_fd,3))
theta_put_fd = -(bt_american('put', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('put', S, X, T-0.001, r, [dividen], [div_time], sigma, N))/0.001
print('Theta of the put option is: ', round(theta_put_fd,3))
rho_call_fd = (bt_american('call', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('call', S, X, T, r - 0.001, [dividen], [div_time], sigma, N))/0.001
print('Rho of the call option is: ', round(rho_call_fd,3))
rho_put_fd = (bt_american('put', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('put', S, X, T, r - 0.001, [dividen], [div_time], sigma, N))/0.001
print('Rho of the put option is: ', round(rho_put_fd,3))
# sensitivity of dividen
divi_call = (bt_american('call', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('call', S, X, T, r, [dividen-0.001], [div_time], sigma, N))/0.001
print('Sensitivity of dividend of the call option is: ', round(divi_call,3))
divi_put = (bt_american('put', S, X, T, r, [dividen], [div_time], sigma, N)-bt_american('put', S, X, T, r, [dividen-0.001], [div_time], sigma, N))/0.001
print('Sensitivity of dividend of the put option is: ', round(divi_put,3))