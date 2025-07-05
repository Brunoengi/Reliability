"""
ANG, Alfredo H.-S.; TANG, Wilson H. Probability concepts in engineering: emphasis on applications in civil and environmental engineering. 2. ed. New York: Wiley, 2007.
Example 4-9, page 161
Expected pf = 0.221
"""
from main import *


def gfunction(x, d):

    g = d[0] * x[0] - (d[1] * x[1] + d[2] * x[2])
    return g

# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation

xvar = [
    {'varname': 'D', 'vardist': 'normal', 'varmean': 1.5, 'varstd': 0.3},
    {'varname': 'A', 'vardist': 'normal', 'varmean': 0.7, 'varstd': 0.2},
    {'varname': 'B', 'vardist': 'normal', 'varmean': 0.5, 'varstd': 0.15},
]
# Design variables

dvar = [
    {'varname': 'factor1', 'varvalue': 1.00},
    {'varname': 'factor2', 'varvalue': 1.00},
    {'varname': 'factor3', 'varvalue': 1.00},
]
#
# MC-IS based on project point
#
storm = Reliability(xvar, dvar, gfunction)
storm.sampling_project_point(100, 5000, 0.005)
