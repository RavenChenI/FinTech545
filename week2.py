import matplotlib
import numpy as np
from scipy.stats import skew, kurtosis, norm
import pandas as pd
import statsmodels.api as sm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import t,probplot
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
# # Problem1
np.random.seed(0)
# 1,000，10,000,100,000 random numbers
random_data = np.random.normal(size=100000)

skew_data = skew(random_data, bias=True)
kurtosis_data = kurtosis(random_data, bias=True)
print(random_data)
print("skew_data by scipy ", skew_data)
print("kurtosis_data  by scipy", kurtosis_data)

skew_data = skew(random_data, bias=False)
kurtosis_data = kurtosis(random_data, bias=False)
print("adjusted_skew_data by scipy ", skew_data)
print("adjusted_kurtosis_data  by scipy", kurtosis_data)
# # ------------------------------------------------------------------

data = pd.read_csv('problem2.csv')
x_data = data['x']
y_data = data['y']
print(type(x_data))
# Adding an Intercept
x_data_with_const = sm.add_constant(x_data)


# ---------------------------------------------
# Problem 2.1
# OLS
ols_model = sm.OLS(y_data, x_data_with_const)
results = ols_model.fit()
print(results.summary())
sns.scatterplot(x=x_data, y=y_data, label="data")
plt.plot(x_data, results.predict(x_data_with_const), color="red", label="Linear Regression")
plt.xlabel = "X"
plt.ylabel = "Y"
plt.legend()
plt.show()
# calculate error vector
predicted_y = results.predict(x_data_with_const)
error_vector = y_data - predicted_y
print(error_vector)
plt.hist(error_vector, bins=20, edgecolor='k')
plt.xlabel = 'Residuals (Errors)'
plt.ylabel = 'Frequency'
plt.title('Distribution of Residuals')
plt.show()
# Q-Q Plot Normal Probability Plot (Q-Q Plot): Create a Q-Q plot of the residuals. In a Q-Q plot, the residuals are
# plotted against theoretical quantiles from a normal distribution. If the residuals closely follow a straight line,
# it suggests that they are approximately normally distributed.
stats.probplot(error_vector, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
# Shapiro-Wilk Test (whether normal distribution)
shapiro_test_statistic, shapiro_p_value = stats.shapiro(error_vector)
print(f'Shapiro-Wilk Test Statistic: {shapiro_test_statistic}')
print(f'p-value: {shapiro_p_value}')
if shapiro_p_value > 0.05:
    print('Residuals may be approximately normally distributed')
else:
    print('Residuals do not appear to be normally distributed')


# ---------------------------------------------------------------------------
# Problem 2.2
# Given the assumption of normality
def MLE_norm(pars, x, y):
    y_est = pars[0] * x + pars[1]
    errors = y - y_est
    sigma = pars[2]
    ll = -np.log(sigma * np.sqrt(2 * np.pi)) - ((errors) ** 2) / (2 * sigma ** 2)
    return -ll.sum()


# Given the assumption of t-distribution

def MLE_t(pars, x, y):
    y_est = pars[0] * x + pars[1]
    errors = y - y_est
    df = pars[2]
    scale = pars[3]
    ll = np.log(t.pdf(errors, df=df, scale=scale))
    return -ll.sum()


pars_norm = [1, 1, 1]

norm_MLE_res = minimize(MLE_norm, pars_norm, args=(x_data, y_data))
norm_Pars = norm_MLE_res.x

pars_t = [1, 0.5, 0.5, 0.5]
t_MLE_res = minimize(MLE_t, pars_t, args=(x_data, y_data))
t_Pars = t_MLE_res.x

print("Given the assumption of normal distribution, MLE parameters are:", norm_Pars[:3])
print("Given the assumption of t-distribution, MLE parameters are:", t_Pars)


def AICs(k, LL, n):
    AIC = 2 * k + 2 * LL
    AICc = AIC + (2 * k * k + 2 * k) / (n - k - 1)

    AICs = []
    AICs.append(AIC)
    AICs.append(AICc)

    return AICs


def BIC(k, LL, n):
    BIC = np.log(n) * k + 2 * LL
    return BIC


print("if Normal distribution, AIC = " + str(AICs(3, norm_MLE_res.fun, len(y_data))[0]),
      " , AICc = " + str(AICs(3, norm_MLE_res.fun, len(y_data))[1]) + " , BIC = " + str(
          BIC(3, norm_MLE_res.fun, len(y_data))))
print("if t distribution, AIC = " + str(AICs(4, t_MLE_res.fun, len(y_data))[0]),
      " , AICc = " + str(AICs(4, norm_MLE_res.fun, len(y_data))[1]) + " , BIC = " + str(BIC(4, norm_MLE_res.fun, len(y_data))))

if AICs(3, norm_MLE_res.fun, len(y_data))[1] > AICs(4, t_MLE_res.fun, len(y_data))[1]:
    print("According to AICc, t distribution performs better")
else:
    print("According to AICc, normal distribution performs better")

if BIC(3, norm_MLE_res.fun, len(y_data)) > BIC(4, t_MLE_res.fun, len(y_data)):
    print("According to BIC, t distribution performs better")
else:
    print("According to BIC, normal distribution performs better")

# -----------------------------------------------------
# Problem 3
def ARMA_plot(title, data, lag):
    data = pd.Series(data)

    with plt.style.context('ggplot'):
        plt.rc('font', size=10)
        fig = plt.figure(figsize=(20, 16))
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        data.plot(ax=ts_ax)
        ts_ax.set_title(title)
        plot_acf(data, lags=lag, ax=acf_ax, alpha=0.05)
        acf_ax.set_ylim(-0.4, 1.1)
        acf_ax.set_title('AutoCovariance Plot')
        plot_pacf(data, lags=lag, ax=pacf_ax, alpha=0.05)
        pacf_ax.set_ylim(-0.4, 1.1)
        pacf_ax.set_title('Partial AutoCovariance Plot')
        sm.qqplot(data, line='s', ax=qq_ax)
        qq_ax.set_title('QQ plot')
        probplot(data, sparams=(data.mean(),
                                data.std()), plot=pp_ax)
        pp_ax.set_title('PP plot')
        plt.tight_layout()
    return


def AR1_process(pars):
    y = []
    y0 = np.random.randn(1)[0]
    y.append(y0)

    for i in range(1, 2001):
        y.append(np.random.randn(1)[0] + pars * y[i - 1])

    title = "AR(1)"
    ARMA_plot(title, y, 30)
    plt.show()


def AR2_process(pars):
    y = []
    y0 = np.random.randn(1)[0]
    y1 = np.random.randn(1)[0]
    y.append(y0)
    y.append(y1)

    for i in range(2, 2001):
        y.append(np.random.randn(1)[0] + pars[0] * y[i - 1] + pars[1] * y[i - 2])
    title = "AR(2)"
    ARMA_plot(title, y, 30)
    plt.show()


def AR3_process(pars):
    y = []
    y0 = np.random.randn(1)[0]
    y1 = np.random.randn(1)[0]
    y.append(y0)
    y.append(y1)

    for i in range(2, 2001):
        y.append(np.random.randn(1)[0] + pars[0] * y[i - 1] + pars[1] * y[i - 2] + pars[2] * y[i - 3])
    title = "AR(3)"
    ARMA_plot(title, y, 30)
    plt.show()

pars=[0.1,0.2,0.3]
AR1_process(pars[0])
AR2_process(pars[:2])
AR3_process(pars)


def MA1_process(pars):
    e = []
    y = []
    y0 = np.random.randn(1)[0]
    e0 = np.random.randn(1)[0]
    e.append(e0)
    y.append(y0)

    Mu = 1
    for i in range(1, 2001):
        tmp_e = np.random.randn(1)[0]
        y.append(tmp_e + Mu + pars * e[i - 1])
        e.append(tmp_e)

    title = "MA(1)"
    ARMA_plot(title, y, 30)

def MA2_process(pars):
    e = []
    y = []
    Mu = 1
    y0 = np.random.randn(1)[0]
    e0 = np.random.randn(1)[0]
    e1 = np.random.randn(1)[0]
    e.append(e0)
    e.append(e1)
    y1 = np.random.randn(1)[0]
    y.append(y0)
    y.append(y1)

    for i in range(2, 2001):
        tmp_e = np.random.randn(1)[0]
        e.append(tmp_e)
        y.append(tmp_e + Mu + pars[0] * e[i - 1] + pars[1] * e[i - 2])

    title = "MA(2)"
    ARMA_plot(title, y, 30)

def MA3_process(pars):
    e = []
    y = []
    Mu = 1
    y0 = np.random.randn(1)[0]
    e0 = np.random.randn(1)[0]
    e1 = np.random.randn(1)[0]
    e2 = np.random.randn(1)[0]
    e.append(e0)
    e.append(e1)
    e.append(e2)
    y1 = np.random.randn(1)[0]
    y2 = np.random.randn(1)[0]
    y.append(y0)
    y.append(y1)
    y.append(y2)
    for i in range(3, 2001):
        tmp_e = np.random.randn(1)[0]
        e.append(tmp_e)
        y.append(tmp_e + Mu + pars[0] * e[i - 1] + pars[1] * e[i - 2] + pars[2] * e[i - 3])

    title = "MA(3)"
    ARMA_plot(title, y, 30)

pars=[0.1,0.2,0.3]

MA1_process(pars[0])
MA2_process(pars[:2])
MA3_process(pars)
plt.show()