"""
Created on Mon Feb 08 17:12:00 2021
Reliabilty Analysis
Example 7.9 - Nonlinear limit state function with non-normal correlated variables
@author: MVREAL
"""
from main import *

#
# Step 0 - Column: g(Y, Z, M) = Y*Z-M = 0
#


def gfunction(x, d):

    g = d[0]*x[0]*x[1]-d[1]*x[2]
    return g


#
# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
    {'varname': 'Y', 'vardist': 'lognormal', 'varmean': 40.00, 'varcov': 0.125},
    {'varname': 'Z', 'vardist': 'lognormal', 'varmean': 50.00, 'varcov': 0.05},
    {'varname': 'M', 'vardist': 'lognormal', 'varmean': 1000.00, 'varstd': 200}
]
# Design variables

dvar = [
    {'varname': 'gamma1', 'varvalue': 1.00},
    {'varname': 'gamma2', 'varvalue': 1.00}
]

# Correlation matrix

corrmatrix = [[1.00, 0.40, 0.00],
              [0.40, 1.00, 0.00],
              [0.00, 0.00, 1.00]]
#
#   MCS  method
#
beam = Reliability(xvar, dvar, gfunction, None, corrmatrix)
beam.mc(100, 5000, 0.01, 1.00)
#