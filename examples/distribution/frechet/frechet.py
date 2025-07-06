from main import *

#
# Step 0 - Beam: g(Y, Z, M) = Y*Z-M = 0
#


def gfunction(x, d):

    g = d[0]*x[0]*x[1]-d[1]*x[2]
    return g


#
# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
    {'varname': 'Y', 'vardist': 'frechet', 'varmean': 38.00, 'varstd': 3},
    {'varname': 'Z', 'vardist': 'frechet', 'varmean': 60.00, 'varcov': 0.05},
    {'varname': 'M', 'vardist': 'frechet', 'varmean': 1000.00, 'varcov': 0.30}
]
# Design variables

dvar = [
    {'varname': 'gamma1', 'varvalue': 1.00},
    {'varname': 'gamma2', 'varvalue': 1.00}
]

#
# MC
#
beam = Reliability(xvar, dvar, gfunction, None, None)
beam.mc(100, 5000, 0.01)
#