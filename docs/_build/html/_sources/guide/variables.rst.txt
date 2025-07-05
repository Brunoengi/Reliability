Variables Distribution 
======================

This section was created to help with the types of random variables and their parameters. This repository works with eight types of random variables, five of which are called usual variables or typical variables and three are called extreme value variables.

Typical Distribution
---------------------------------------------------

- `Normal Distribution <#id1>`_
- `Log-normal Distribution <#id2>`_
- `Uniform Distribution <#id3>`_
- `Beta Distribution <#id4>`_
- `Gamma Distribution <#id5>`_


Extreme Value Distribution
---------------------------------------------------
- `Gumbel Distribution <#id6>`_
- `Frechet Distribution <#id7>`_
- `Weibull Distribution <#id8>`_

How to create a variable set
--------------------------------------------------

Variables must be entered in the form of a list where each position in the list will be a dictionary. In the following example, we have a problem involving three variables.

  .. code-block:: bash

    xvar = [
      {'varname': 'Y', 'vardist': 'lognormal', 'varmean': 38.00, 'varcov': 0.10},
      {'varname': 'Z', 'vardist': 'normal', 'varmean': 60.00, 'varcov': 0.05},
      {'varname': 'M', 'vardist': 'frechet', 'varmean': 1000.00, 'varcov': 0.30}
    ]

However, the common mandatory parameters of all probability distributions are:
``varname``,
``vardist``

The other dictionary keys depend on each distribution type. In this case, it is necessary to read the documentation of the distribution of interest.

List of all valid attributes and their expected data types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

=========================  =========================
Key                         Type (Value)
=========================  =========================
varname                       string
vardist                       'normal' | 'lognormal' | 'uniform' | 'gumbel' | 'frechet' | 'weibull' | 'beta' | 'gama'  
varmean                       number | float
varcov                        number | float
varstd                        number | float
varhmean                      number | float
parameter1                    number | float
parameter2                    number | float
parameter3                    number | float
parameter4                    number | float
=========================  =========================

Normal Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Possible keys:
                
================ ======================================  =========================
Combination          Required                                           Optional
================ ======================================  =========================
1                  varname, vardist, varmean, varcov                      varhmean
2                  varname, vardist, varmean, varstd                      varhmean
================ ======================================  =========================

Log-normal Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Possible keys:
                
================ ======================================  =========================
Combination          Required                                           Optional
================ ======================================  =========================
1                  varname, vardist, varmean, varcov                      varhmean
2                  varname, vardist, varmean, varstd                      varhmean
================ ======================================  =========================

Uniform Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Possible keys:
                
================ ===============================================  =========================
Combination          Required                                           Optional
================ ===============================================  =========================
1                  varname, vardist, varmean, varstd                      varhmean
2                  varname, vardist, parameter1, parameter2               varhmean
================ ===============================================  =========================

Beta Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Possible keys:
                
================ ===============================================================================  =========================
Combination          Required                                                                             Optional
================ ===============================================================================  =========================
1                  varname, vardist, parameter1, parameter2, parameter3, parameter4                       varhmean
================ ===============================================================================  =========================

Gamma Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Possible keys:
                
================ ======================================  =========================
Combination          Required                                           Optional
================ ======================================  =========================
1                  varname, vardist, varmean, varstd                      varhmean
2                  varname, vardist, varmean, varcov                      varhmean
================ ======================================  =========================


Gumbel Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

================ ======================================  =========================
Combination          Required                                           Optional
================ ======================================  =========================
1                  varname, vardist, varmean, varstd                      varhmean
2                  varname, vardist, varmean, varcov                      varhmean
================ ======================================  =========================


Frechet Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================ ======================================  =========================
Combination          Required                                           Optional
================ ======================================  =========================
1                  varname, vardist, varmean, varcov                      varhmean
================ ======================================  =========================

Weibull Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================ ===============================================  =========================
Combination          Required                                           Optional
================ ===============================================  =========================
1                varname, vardist, varmean, varstd, parameter3          varhmean
================ ===============================================  =========================
