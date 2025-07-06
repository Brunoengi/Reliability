"""
ANG, Alfredo H.-S.; TANG, Wilson H. Probability concepts in engineering: emphasis on applications in civil and environmental engineering. 2. ed. New York: Wiley, 2007.
Example 5-10, page 218
Expected pf = 0.202
"""
from main import *
import math

def gfunction(x, d):

    g = d[0] * x[0] - (d[1] * x[1] + d[2] * x[2])
    return g

# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
    {'varname': 'D', 'vardist': 'normal', 'varmean': 1.5, 'varstd': 0.3},
    {'varname': 'A', 'vardist': 'gamma', 'varmean': 12/17.5, 'varstd': math.sqrt(12/17.5**2)},
    {'varname': 'B', 'vardist': 'gamma', 'varmean': 11/22.22, 'varstd': math.sqrt(11/22.22**2)},
]
# Design variables

dvar = [
    {'varname': 'factor1', 'varvalue': 1.00},
    {'varname': 'factor2', 'varvalue': 1.00},
    {'varname': 'factor3', 'varvalue': 1.00},
]
#
# MCS enhanced method
#
storm = Reliability(xvar, dvar, gfunction)
storm.sampling_enhanced(100, 5000, 0.005)

