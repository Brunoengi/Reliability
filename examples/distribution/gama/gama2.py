from main import *

def gfunction(x, d):
    g = d[0] * x[0] - d[1] * x[1]
    return g

# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
    {'varname': 'A', 'vardist': 'gamma', 'varmean': 20, 'varstd': 5},
    {'varname': 'B', 'vardist': 'gamma', 'varmean': 10, 'varcov': 0.3},
]
# Design variables

dvar = [
    {'varname': 'factor1', 'varvalue': 1.00},
    {'varname': 'factor2', 'varvalue': 1.00},
]

corrmatrix = [[1, 0.2],
              [0.2, 1]]

# MCS method
#
test = Reliability(xvar, dvar, gfunction, None, corrmatrix)
test.mc(100, 5000, 0.005)

