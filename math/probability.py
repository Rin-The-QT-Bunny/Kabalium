"""
Distributions and Probability Tools
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
#plt.style.use('seaborn')

def C(n,m):
    # Cnm means the combinational number of n boxes with m balls
    return math.factorial(n)//(math.factorial(m)*math.factorial(n-m))

def binomial(p,n,k):
    # out put the binomial distribution
    # k number events happened in n individuals with probability of p
    return C(n,k) * p ** k * (1-p)**(n-k)


# 第一簇的数据
num1, mu1, var1 = 400, [0.5, 0.5], [1, 3]
X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
# 第二簇的数据
num2, mu2, var2 = 600, [5.5, 2.5], [2, 2]
X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
# 第三簇的数据
num3, mu3, var3 = 1000, [1, 7], [6, 2]
X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
# 合并在一起
X = np.vstack((X1, X2, X3))
