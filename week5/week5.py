import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import qfunctions as qf
#Problem 1
#Use the data in problem1.csv.  Fit a Normal Distribution and a Generalized T distribution to this data.
#Calculate the VaR and ES for both fitted distributions.

data = pd.read_csv('problem1.csv')
# Fit a normal distribution to the data:
mu, std = norm.fit(data)
# Fit a t distribution to the data:
df, loc, scale = t.fit(data)
n = 10000
normal_data = norm.rvs(mu, std, size=n)
t_data = t.rvs(df, loc, scale, size=n)
# Calculate VaR for both distributions:
normal_var = qf.calculate_var(normal_data)
t_var = qf.calculate_var(t_data)
def calculate_es(data, mean = 0, alpha=0.05):
    return -np.mean(data[data<-qf.calculate_var(data, mean, alpha)])
# Calculate ES for both distributions:
normal_es = calculate_es(normal_data)
t_es = calculate_es(t_data)
xnorm = np.linspace(-0.5, 0.5, num=100)
pdf = norm.pdf(xnorm, mu, std)

xnorm = np.linspace(-0.5, 0.5, num=100)
pdft = t.pdf(xnorm, df, loc, scale)
plt.figure(figsize=(10,6))
plt.hist(data, bins=100, density=True, alpha=0.5, label='Origin data')
plt.hist(normal_data, bins=100, density=True, alpha=0.5, label='Normal')
plt.hist(t_data, bins=100, density=True, alpha=0.5, label='T')
plt.plot(xnorm, pdf, label = 'Normal Distribution')
plt.plot(xnorm, pdft, label = 'T Distribution')
plt.axvline(-normal_var, color='r', linestyle='--', label='Normal VaR')
plt.axvline(-t_var, color='g', linestyle='--', label='T VaR')
plt.axvline(-normal_es, color='r', linestyle='-', label='Normal ES')
plt.axvline(-t_es, color='g', linestyle='-', label='T ES')
plt.legend()
plt.show()

#Problem 2
data = pd.read_csv("DailyPrices.csv", index_col=0)
#Covariance estimation techniques.
weights = qf.weight_gen(len(data.iloc[:,0]), 0.94)
print(weights.shape)
cov = qf.ewcov_gen(data, weights)
print(cov.shape)
#Non PSD fixes for correlation matrices
n = 500
sigma = np.matrix(np.full((n, n), 0.9))
np.fill_diagonal(sigma, 1)
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357

near_psd_matrix = qf.near_psd(sigma)
print(qf.is_psd(near_psd_matrix))
higham_psd_matrix = qf.Higham_method(sigma)
print(qf.is_psd(higham_psd_matrix))
#Simulation Methods
direct_sim = qf.sim_mvn_from_cov(cov)
print(direct_sim.shape)
pca_sim = qf.PCA_with_percent(cov)
print(pca_sim.shape)
# VaR calculation methods (all discussed)
samplePrice = data.iloc[:,0]
samplereturn = qf.return_calculate(samplePrice)
samplereturn = samplereturn - np.mean(samplereturn)

print(qf.calculate_var(samplereturn))
print(qf.normal_var(samplereturn))
print(qf.ewcov_normal_var(samplereturn))
print(qf.t_var(samplereturn))
print(qf.historic_var(samplereturn))

#ES calculation
es = qf.calculate_es(samplereturn)
print(es)

# Problem 3
portfolio = pd.read_csv('portfolio.csv')
dailyPrice = pd.read_csv("DailyPrices.csv", index_col=0)
portfolioA = portfolio[portfolio['Portfolio'] == 'A']
portfolioB = portfolio[portfolio['Portfolio'] == 'B']
portfolioC = portfolio[portfolio['Portfolio'] == 'C']
returns = dailyPrice.pct_change().dropna(how='all')
returns = returns - np.mean(returns)
#calculate PV
def portfolio_price(dailyprice, portfolio):
    pv = []
    for stock in portfolio['Stock']:
        pv.append(dailyprice.iloc[-1][stock])
    return pv
def get_return(port):
    returns_p=[]
    for stock in port.loc[:, 'Stock'].tolist():
        returns_p.append((returns.loc[:, stock]).tolist())
    returns_p=pd.DataFrame(returns_p).T
    return returns_p

print(get_return(portfolioA))

def cal_t_pVals(port, returns_port, price):
    return_cdf=[]
    par=[]
    for col in returns_port.columns:
        df, loc, scale = t.fit(returns_port[col].values)
        par.append([df,loc,scale])
        return_cdf.append(t.cdf(returns_port[col].values, df=df, loc=loc, scale=scale).tolist())
    return_cdf=pd.DataFrame(return_cdf).T
    spearman_cor=return_cdf.corr(method='spearman')
    sample=pd.DataFrame(qf.PCA_with_percent(spearman_cor)).T
    sample_cdf=[]
    for col in sample.columns:
        sample_cdf.append(norm.cdf(sample[col].values, loc=0, scale=1).tolist())
    simu_return=[]
    for i in range(len(sample_cdf)):
        simu_return.append(t.ppf(sample_cdf[i], df=par[i][0], loc=par[i][1], scale=par[i][2]))
    simu_return=np.array(simu_return)

    sim_price=(1 + simu_return.T)*price
    pVals = sim_price.dot(port['Holding'])
    pVals.sort()
    return pVals

print(cal_t_pVals(portfolioA, get_return(portfolioA), portfolio_price(dailyPrice, portfolioA)))

pv = np.sum(portfolioA['Holding']*portfolio_price(dailyPrice, portfolioA))
print("portfolio A")
print("VaR: ", qf.calculate_var(cal_t_pVals(portfolioA, get_return(portfolioA), portfolio_price(dailyPrice, portfolioA)), pv))
print("ES: ", qf.calculate_es(pv-cal_t_pVals(portfolioA, get_return(portfolioA), portfolio_price(dailyPrice, portfolioA))))

pv = np.sum(portfolioB['Holding']*portfolio_price(dailyPrice, portfolioB))
print("portfolio B")
print("VaR: ", qf.calculate_var(cal_t_pVals(portfolioB, get_return(portfolioB), portfolio_price(dailyPrice, portfolioB)), pv))
print("ES: ", qf.calculate_es(pv-cal_t_pVals(portfolioB, get_return(portfolioB), portfolio_price(dailyPrice, portfolioB))))


pv = np.sum(portfolioC['Holding']*portfolio_price(dailyPrice, portfolioC))
print("portfolio C")
print("VaR: ", qf.calculate_var(cal_t_pVals(portfolioC, get_return(portfolioC), portfolio_price(dailyPrice, portfolioC)), pv))
print("ES: ", qf.calculate_es(pv-cal_t_pVals(portfolioC, get_return(portfolioC), portfolio_price(dailyPrice, portfolioC))))

pv = np.sum(portfolio['Holding']*portfolio_price(dailyPrice, portfolio))
print("portfolio Total")
print("VaR: ", qf.calculate_var(cal_t_pVals(portfolio, get_return(portfolio), portfolio_price(dailyPrice, portfolio)), pv))
print("ES: ", qf.calculate_es(pv-cal_t_pVals(portfolio, get_return(portfolio), portfolio_price(dailyPrice, portfolio))))