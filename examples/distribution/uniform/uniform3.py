# Expected pf = 0.125

from main import *

def gfunction(x, d):

    g = d[0] * x[0] - d[1] * x[1]
    return g

# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
  {'varname': 'R', 'vardist': 'uniform', 'varmean': 15, 'varstd': 2.88675},
  {'varname': 'S', 'vardist': 'uniform', 'varmean': 10, 'varstd': 2.88675},
]

# Design variables

dvar = [
  {'varname': 'factor1', 'varvalue': 1.00},
  {'varname': 'factor2', 'varvalue': 1.00},
]
#
# MCS method
#
test = Reliability(xvar, dvar, gfunction)
test.mc(100, 5000, 0.005)
