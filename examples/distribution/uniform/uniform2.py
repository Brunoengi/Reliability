from main import *

def gfunction(x, d):

    g = d[0] * x[0] - d[1] * x[1]
    return g

# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
  {'varname': 'R', 'vardist': 'uniform', 'parameter1': 10, 'parameter2': 20},
  {'varname': 'S', 'vardist': 'uniform', 'parameter1': 5, 'parameter2': 15},
]

# Design variables

dvar = [
  {'varname': 'factor1', 'varvalue': 1.00},
  {'varname': 'factor2', 'varvalue': 1.00},
]

# Correlation matrix
corrmatrix = [[1.00, 0.50],
              [0.50, 1.00]]
#
# MCS method
#
test = Reliability(xvar, dvar, gfunction, None, corrmatrix)
test.mc(100, 5000, 0.005)
