import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

data = pd.read_csv('DailyReturn.csv', index_col=0)


# Problem1
def weight_gen(n, lambd):
    weight = np.zeros(n)
    for i in range(n):
        weight[i] = (1 - lambd) * lambd ** i
    z_weight = weight / np.sum(weight)
    return z_weight[::-1]


def cov_gen(data, weight):
    data = data - data.mean()
    weight = np.diag(weight)
    left = weight @ data
    right = np.dot(data.T, left)
    return right


def PCA(cov):
    eigenvalue, eigenvector = np.linalg.eig(cov)
    # descending order
    z_list = list(zip(eigenvalue, eigenvector))
    z_list.sort(key=lambda x: x[0], reverse=True)
    eigenvalue = [x[0] for x in z_list]
    eigenvector = [x[1] for x in z_list]
    # calculate the cumulative variance
    explanation = eigenvalue / np.sum(eigenvalue)
    cumulative = np.cumsum(explanation)
    cumulative[-1] = 1
    return cumulative


lambd = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98]
plt.figure(figsize=(8, 6))
for i, lambd in enumerate(lambd):
    cumulative = PCA(cov_gen(data, weight_gen(data.shape[0], lambd)))
    plt.plot(np.arange(0, data.shape[1]), cumulative, label=f"lambda = {lambd}")

plt.xlabel("number of eigenvalues")
plt.ylabel("variance explained")
plt.legend()
plt.show()

# Problem2
n = 6
Higham = np.zeros((n, n))
# make all value in Highim matrix 0.9
Higham.fill(0.9)
for i in range(n):
    Higham[i, i] = 1
Higham[0, 1] = 0.7357
Higham[1, 0] = 0.7357


# implement chol_psd()
def chol_psd(a):
    n = a.shape[1]
    root = np.zeros((n, n))
    for j in range(n):
        s = 0
        if j > 0:
            s = root[j, 0:j].T @ root[j, 0:j]
        temp = a[j, j] - s
        if 0 >= temp >= -1e-3:
            temp = 0
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0:
            for i in range(j, n):
                root[j, i] = 0
        else:
            ir = 1 / root[j, j]
            for i in range(j + 1, n):
                s = root[i, 0:j].T @ root[j, 0:j]
                root[i, j] = (a[i, j] - s) * ir
    return root


# root = chol_psd(Higham)
# print(root)
# print()
# print(root @ root.T)


# implement near_psd()
def near_psd(a, epsilon=0.0):
    is_cov = False
    for i in np.diag(a):
        if abs(i - 1) > 1e-8:
            is_cov = True
        else:
            is_cov = False
            break
    if is_cov:
        invSD = np.diag(1 / np.sqrt(np.diag(a)))
        a = invSD @ a @ invSD
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i, epsilon) for i in vals])
    T = 1 / (np.square(vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    if is_cov:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out


root = near_psd(Higham)


# Implement Higham’s 2002 nearest psd correlation function
def Frobenius_Norm(a):
    return np.sqrt(np.sum(np.square(a)))


def projection_u(a):
    np.fill_diagonal(a, 1.0)
    return a


def projection_s(a, epsilon=1e-7):
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i, epsilon) for i in vals])
    return vecs @ np.diag(vals) @ vecs.T


def Higham_method(a, tol=1e-8):
    s = 0
    gamma = np.inf
    y = a
    # iteration
    while True:
        r = y - s
        x = projection_s(r)
        s = x - r
        y = projection_u(x)
        gamma_next = Frobenius_Norm(y - a)
        if abs(gamma - gamma_next) < tol:
            break
        gamma = gamma_next
    return y


n = 500
Higham_500 = np.zeros((n, n))
# make all value in Highim matrix 0.9
Higham_500.fill(0.9)
for i in range(n):
    Higham_500[i, i] = 1
Higham_500[0, 1] = 0.7357
Higham_500[1, 0] = 0.7357


def is_psd(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)


print(is_psd(near_psd(Higham_500)))
print(is_psd(Higham_method(Higham_500)))


def Higham_with_n(n):
    Higham = np.zeros((n, n))
    # make all value in Highim matrix 0.9
    Higham.fill(0.9)
    for i in range(n):
        Higham[i, i] = 1
    Higham[0, 1] = 0.7357
    Higham[1, 0] = 0.7357
    return Higham


n = [100, 300, 500, 700, 1000]
Higham_method_time = []
Near_psd_time = []
for i in n:
    Higham = Higham_with_n(i)
    Higham_method_time.append(timeit.timeit(lambda: Higham_method(Higham), number=1))
    Near_psd_time.append(timeit.timeit(lambda: near_psd(Higham), number=1))

plt.figure(figsize=(8, 6))
plt.plot(n, Higham_method_time, label="Higham_method")
plt.plot(n, Near_psd_time, label="Near_psd")
plt.legend()
plt.xlabel("n")
plt.ylabel("Time")
plt.show()

Higham_method_frob = []
Near_psd_frob = []
for i in n:
    Higham = Higham_with_n(i)
    Higham_method_frob.append(Frobenius_Norm(Higham_method(Higham) - Higham))
    Near_psd_frob.append(Frobenius_Norm(near_psd(Higham) - Higham))
plt.figure(figsize=(8, 6))
plt.plot(n, Higham_method_frob, label="Higham_method")
plt.plot(n, Near_psd_frob, label="Near_psd")
plt.legend()
plt.xlabel("n")
plt.ylabel("Frobeinuis Norm")
plt.show()


# Problem3
def multivariate_normal_distribution(cov, num_of_simulation=30000):
    return chol_psd(cov) @ np.random.normal(size=(cov.shape[0], num_of_simulation))


def var(cov):
    return np.diag(cov)


def corr(cov):
    return np.diag(1 / np.sqrt(var(cov))) @ cov @ np.diag(1 / np.sqrt(var(cov))).T


def cov(var, cor):
    std = np.sqrt(var)
    return np.diag(std) @ cor @ np.diag(std).T


ew_cov = cov_gen(data, weight_gen(data.shape[0], 0.97))
normal_cov = np.cov(data.T)
norm_corr_normal_var = cov(var(normal_cov), corr(normal_cov))
ew_corr_normal_var = cov(var(normal_cov), corr(ew_cov))
normal_corr_ew_var = cov(var(ew_cov), corr(normal_cov))
ew_corr_ew_var = cov(var(ew_cov), corr(ew_cov))


def PCA_with_percent(cov, percent=0.95, num_of_simulation=25000):
    eigenvalue, eigenvector = np.linalg.eigh(cov)
    total = np.sum(eigenvalue)
    for i in range(cov.shape[0]):
        i = len(eigenvalue) - i - 1
        if eigenvalue[i] < 0:
            eigenvalue = eigenvalue[i + 1:]
            eigenvector = eigenvector[:, i + 1:]
            break
        if sum(eigenvalue[i:]) / total > percent:
            eigenvalue = eigenvalue[i:]
            eigenvector = eigenvector[:, i:]
            break
    simulate = np.random.normal(size=(len(eigenvalue), num_of_simulation))
    return eigenvector @ np.diag(np.sqrt(eigenvalue)) @ simulate


covs = [norm_corr_normal_var, ew_corr_normal_var, normal_corr_ew_var, ew_corr_ew_var]
title = ["norm_corr_normal_var", "ew_corr_normal_var", "normal_corr_ew_var", "ew_corr_ew_var"]
for i in range(len(covs)):
    plt.figure(i, figsize=(8, 6))
    plt.title(title[i])
    cov = covs[i]
    data1 = multivariate_normal_distribution(cov, 25000)
    data2 = PCA_with_percent(cov, 1, 25000)
    data3 = PCA_with_percent(cov, 0.75, 25000)
    data4 = PCA_with_percent(cov, 0.5, 25000)
    plt.bar("Direct Simulation", Frobenius_Norm(cov - np.cov(data1)))
    plt.bar("PCA with 100%", Frobenius_Norm(cov - np.cov(data2)))
    plt.bar("PCA with 75%", Frobenius_Norm(cov - np.cov(data3)))
    plt.bar("PCA with 50%", Frobenius_Norm(cov - np.cov(data4)))
    plt.show()



# Compare runtime
import time
covs = [norm_corr_normal_var, ew_corr_normal_var, normal_corr_ew_var, ew_corr_ew_var]
title = ["norm_corr_normal_var", "ew_corr_normal_var", "normal_corr_ew_var", "ew_corr_ew_var"]
for i in range(len(covs)):
    plt.figure(i,figsize=(8,6))
    plt.title(title[i])
    cov = covs[i]
    start = time.time()
    time_data1 = multivariate_normal_distribution(cov,25000)
    end1 = time.time() - start
    start = time.time()
    time_data2 = PCA_with_percent(cov, 1, 25000)
    end2 = time.time() - start
    start = time.time()
    time_data3 = PCA_with_percent(cov, 0.75, 25000)
    end3 = time.time() - start
    start = time.time()
    time_data4 = PCA_with_percent(cov, 0.5, 25000)
    end4 = time.time() - start
    plt.bar("Direct Simulation",end1)
    plt.bar("PCA with 100%",end2)
    plt.bar("PCA with 75%",end3)
    plt.bar("PCA with 50%",end4)
    plt.show()