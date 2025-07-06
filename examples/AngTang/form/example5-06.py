"""
ANG, Alfredo H.-S.; TANG, Wilson H. Probability concepts in engineering: emphasis on applications in civil and environmental engineering. 2. ed. New York: Wiley, 2007.
Example 5-06, page 209
Expected pf = 1 - 0.721 = 0.279
"""

from main import *

def gfunction(x, d):

    X1 = d[0] * x[0] 
    X2 = d[1] * x[1]
    X3 = d[2] * x[2]
    X4 = d[3] * x[3]
    X5 = d[4] * x[4]

    T1 = X1 + X2 + X3
    T2 = X4 + X5

    g = 8 - max(T1,T2)

    return g

# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
    {'varname': '1-2', 'vardist': 'beta', 'varmean': 2, 'parameter1': 0, 'parameter2': 5.0, 'parameter3': 2.0, 'parameter4': 3.0},
    {'varname': '2-3', 'vardist': 'beta', 'varmean': 1, 'parameter1': 0, 'parameter2': 2.5, 'parameter3': 2.0, 'parameter4': 3.0},
    {'varname': '3-5', 'vardist': 'beta', 'varmean': 3, 'parameter1': 1, 'parameter2': 6.0, 'parameter3': 2.0, 'parameter4': 3.0},
    {'varname': '1-4', 'vardist': 'beta', 'varmean': 5, 'parameter1': 3, 'parameter2': 8.0, 'parameter3': 2.0, 'parameter4': 3.0},
    {'varname': '4-5', 'vardist': 'beta', 'varmean': 2, 'parameter1': 1, 'parameter2': 3.5, 'parameter3': 2.0, 'parameter4': 3.0},
]

# Design variables

dvar = [
    {'varname': 'factor1', 'varvalue': 1.00},
    {'varname': 'factor2', 'varvalue': 1.00},
    {'varname': 'factor3', 'varvalue': 1.00},
    {'varname': 'factor4', 'varvalue': 1.00},
    {'varname': 'factor5', 'varvalue': 1.00}
]

#
# FORM method
#
construction = Reliability(xvar, dvar, gfunction, None, None)
construction.form(iHLRF=True, toler=1.e-6)
