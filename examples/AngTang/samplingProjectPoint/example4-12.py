"""
Created on Mon Feb 08 17:12:00 2021
Reliabilty Analysis
Example 7.3 - Linear limit state function with normal independent variables
@author: MVREAL

ANG, Alfredo H.-S.; TANG, Wilson H. Probability concepts in engineering: emphasis on applications in civil and environmental engineering. 2. ed. New York: Wiley, 2007.
Example 4-12, page 164 
Expected pf = 1 - 0.74 = 0.26
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
    {'varname': '1-2', 'vardist': 'normal', 'varmean': 2, 'varstd': 1},
    {'varname': '2-3', 'vardist': 'normal', 'varmean': 1, 'varstd': 0.5},
    {'varname': '3-5', 'vardist': 'normal', 'varmean': 3, 'varstd': 1},
    {'varname': '1-4', 'vardist': 'normal', 'varmean': 5, 'varstd': 1},
    {'varname': '4-5', 'vardist': 'normal', 'varmean': 2, 'varstd': 0.5}
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
# MC-IS based on project point
#
construction = Reliability(xvar, dvar, gfunction, None, None)
construction.sampling_project_point(100, 5000, 0.005)