"""
Created on Mon Feb 08 17:12:00 2021
Reliabilty Analysis
Example 7.3 - Linear limit state function with normal independent variables
@author: MVREAL

ANG, Alfredo H.-S.; TANG, Wilson H. Probability concepts in engineering: emphasis on applications in civil and environmental engineering. 2. ed. New York: Wiley, 2007.
Example 4-11, page 163 - Adapted
Expected pf = 0.0174
"""
from main import *

#
# Step 0 - Column: g(R, G, Q, W) = R-G-Q-W = 0
#


def gfunction(x, d):

    g = d[0] * x[0] - d[1] * x[1] - d[2] * x[2] - d[3] * x[3]
    return g



# Data input
#
# Random variables: name, probability distribution, mean and coefficient of variation


xvar = [
    {'varname': 'R', 'vardist': 'normal', 'varmean': 975.00, 'varcov': 0.15},
    {'varname': 'G', 'vardist': 'normal', 'varmean': 200.00, 'varcov': 0.07},
    {'varname': 'Q', 'vardist': 'normal', 'varmean': 300.00, 'varcov': 0.12},
    {'varname': 'w', 'vardist': 'normal', 'varmean': 150.00, 'varcov': 0.20}
]
# Design variables

dvar = [
    {'varname': 'factor1', 'varvalue': 1.00},
    {'varname': 'factor2', 'varvalue': 1.00},
    {'varname': 'factor3', 'varvalue': 1.00},
    {'varname': 'factor4', 'varvalue': 1.00}
]

# MCS  method
#
column = Reliability(xvar, dvar, gfunction)
column.mc(1000, 10000, 0.01)
#