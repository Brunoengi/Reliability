from main import *

def gfunction(x, d):

    g = d[0] * x[0] - d[1] * x[1]
    return g

# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
  {'varname': 'R', 'vardist': 'weibull', 'varmean': 100, 'varstd': 20, 'parameter3':10},
  {'varname': 'S', 'vardist': 'weibull', 'varmean': 100, 'varstd': 20, 'parameter3':10},
]

# Design variables

dvar = [
  {'varname': 'factor1', 'varvalue': 1.00},
  {'varname': 'factor2', 'varvalue': 1.00},
]

corrmatrix = [[1, 0.5],
              [0.5, 1]]

#
# MCS method
#
test = Reliability(xvar, dvar, gfunction, None, corrmatrix)
test.mc(100, 5000, 0.005)
