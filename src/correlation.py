import numpy as np

class Correlation:

  def __init__(self, props):

    """
      Initializes an instance of the class.

      Parameters
      ----------
      props : object
          An object (such as a configuration class or a namespace) that must contain
          the following attributes:

          - Rz : array_like
              Correlation matrix without nataf correction.
          - Rz_rectify : array_like
              Correlation matrix with nataf correction.  
          - xvar : list or array_like
              List or array of the random variables in the problem.
          - nxvar : int
              Number of random variables in xvar.

      Notes
      -----
      The `props` object must provide all the required attributes listed above.
      Otherwise, an `AttributeError` will be raised during initialization.

      Internal Attributes
      -------------------
      tolerance : float
          Numerical tolerance used internally (default value is 1e-10).
      """

    self.Rz = props['Rz']
    self.xvar = props['xvar']
    self.nxvar = props['nxvar']
    self.tolerance = 1e-10

    if self.is_identity_matrix():
            self.Rz_rectify = self.Rz.copy()
    else:
        self.Rz_rectify = self.nataf()
            
  def correlation_summary(self):
    """
    Returns two lists:
    - indices of variables that are correlated with at least one other variable
    - indices of variables that are uncorrelated with all other variables

    Parameters:
    - R: Correlation matrix (must be square)
    - tol: Tolerance threshold to consider a correlation as nonzero
    """
    n = self.Rz.shape[0]
    correlated = set()

    for i in range(n):
        for j in range(n):
            if i != j and abs(self.Rz[i, j]) > self.tolerance:
                correlated.add(i)
                break  # i is correlated with at least one j

    all_indices = set(range(n))
    uncorrelated = all_indices - correlated

    return sorted(correlated), sorted(uncorrelated)
  
  def is_identity_matrix(self):
    # Creates an identity matrix
    I = np.eye(self.Rz.shape[0])
    # Compare with tolerance to deal with numerical errors
    return np.allclose(self.Rz, I, atol=self.tolerance)

  def nataf(self):
        """
        Nataf correction of the correlation matrix
        According to:
        Liu, P.-L. and Kiureghian, A.D. Multivariate distribution models with prescribed marginals and covariances
        Probabilistic Engineering Mechanics, 1986, Vol. 1, No.2, p. 105-112
        """
        Rz1 = np.array(self.Rz)
        for i in range(self.nxvar):
            for j in range(i):
                # Variables parameters
                f = 1.00
                ro = self.Rz[i][j]
                cvi = float(self.xvar[i].varcov)
                cvj = float(self.xvar[j].varcov)

                # Table 4: Xi is gauss and Xj belongs to group 1 - f is constant

                # 1 Xi = gauss and Xj = gauss

                if self.xvar[i].vardist == 'gauss' and self.xvar[j].vardist == 'gauss':
                    f = 1.000

                # 2 Xi = gauss and Xj = uniform

                elif self.xvar[i].vardist == 'gauss' and self.xvar[j].vardist == 'uniform' \
                        or self.xvar[i].vardist == 'uniform' and self.xvar[j].vardist == 'gauss':
                    f = 1.023

                # 3 Xi = gauss and Xj = gumbel

                elif self.xvar[i].vardist == 'gauss' and self.xvar[j].vardist == 'gumbel' \
                        or self.xvar[i].vardist == 'gumbel' and self.xvar[j].vardist == 'gauss':
                    f = 1.031

                # Table 5: Xi is gauss and Xj belongs to group 2 - f depends on cvj

                # 4 Xi = gauss and Xj = lognorm

                elif self.xvar[i].vardist == 'gauss' and self.xvar[j].vardist == 'lognorm' \
                        or self.xvar[i].vardist == 'lognorm' and self.xvar[j].vardist == 'gauss':
                    if self.xvar[i].vardist == 'lognorm':
                        cv = cvi
                    else:
                        cv = cvj
                    f = cv / (np.sqrt(np.log(1.00 + cv ** 2)))

                # 5 Xi = gauss and Xj = frechet

                elif self.xvar[i].vardist == 'gauss' and self.xvar[j].vardist == 'frechet' \
                        or self.xvar[i].vardist == 'frechet' and self.xvar[j].vardist == 'gauss':
                    if self.xvar[i].vardist == 'frechet':
                        cv = cvi
                    else:
                        cv = cvj
                    f = 1.030 + 0.238 * cv + 0.364 * cv ** 2

                # 6 Xi = gauss and Xj = weibull - min

                elif self.xvar[i].vardist == 'gauss' and self.xvar[i].vardist == 'weibull' \
                        or self.xvar[i].vardist == 'weibull' and self.xvar[j].vardist == 'gauss':
                    if self.xvar[i].vardist == 'weibull':
                        cv = cvi
                    else:
                        cv = cvj
                    f = 1.031 - 0.195 * cv + 0.328 * cv ** 2

                # Table 6: Xi  and Xj belongs to group 2 - f depends on ro

                # 7 Xi = uniform and Xj = uniform

                elif self.xvar[i].vardist == 'uniform' and self.xvar[j].vardist == 'uniform':
                    f = 1.047 - 0.047 * ro ** 2

                # 8 Xi = gumbel and Xj = gumbel

                elif self.xvar[i].vardist == 'gumbel' and self.xvar[j].vardist == 'gumbel':
                    f = 1.064 - 0.069 * ro + 0.005 * ro ** 2

                # 9 Xi = uniform and Xj = gumbel

                elif self.xvar[i].vardist == 'uniform' and self.xvar[j].vardist == 'gumbel' \
                        or self.xvar[i].vardist == 'gumbel' and self.xvar[j].vardist == 'uniform':
                    f = 1.055 + 0.015 * ro ** 2

                # Table 7: Xi belongs to group 1 and Xj belongs to group 2 - f depends on ro and cvj

                # 10 Xi = uniform and Xj = lognorm

                elif self.xvar[i].vardist == 'uniform' and self.xvar[j].vardist == 'lognorm' \
                        or self.xvar[i].vardist == 'lognorm' and self.xvar[j].vardist == 'uniform':
                    if self.xvar[i].vardist == 'lognorm':
                        cv = cvi
                    else:
                        cv = cvj
                    f = 1.019 - 0.014 * cv + 0.010 * ro ** 2 + 0.249 * cv ** 2

                # 11 Xi = uniform and Xj = frechet

                elif self.xvar[i].vardist == 'uniform' and self.xvar[j].vardist == 'frechet' \
                        or self.xvar[i].vardist == 'frechet' and self.xvar[j].vardist == 'uniform':
                    if self.xvar[i].vardist == 'frechet':
                        cv = cvi
                    else:
                        cv = cvj
                    f = 1.033 + 0.305 * cv + 0.074 * ro ** 2 + 0.405 * cv ** 2

                # 12 Xi = uniform and Xj = weibull - min

                elif self.xvar[i].vardist == 'uniform' and self.xvar[j].vardist == 'weibull' \
                        or self.xvar[i].vardist == 'weibull' and self.xvar[j].vardist == 'uniform':
                    if self.xvar[i].vardist == 'weibull':
                        cv = cvi
                    else:
                        cv = cvj
                    f = 1.061 - 0.237 * cv - 0.005 * ro ** 2 + 0.379 * cv ** 2

                # 13 Xi = gumbel and Xj = lognorm

                elif self.xvar[i].vardist == 'gumbel' and self.xvar[j].vardist == 'lognorm' \
                        or self.xvar[i].vardist == 'lognorm' and self.xvar[j].vardist == 'gumbel':
                    if self.xvar[i].vardist == 'lognorm':
                        cv = cvi
                    else:
                        cv = cvj
                    f = 1.029 + 0.001 * ro + 0.014 * cv + 0.004 * ro ** 2 + 0.233 * cv ** 2 - 0.197 * ro * cv

                # 14 Xi = gumbel and Xj = frechet

                elif self.xvar[i].vardist == 'gumbel' and self.xvar[j].vardist == 'frechet' \
                        or self.xvar[i].vardist == 'frechet' and self.xvar[j].vardist == 'gumbel':
                    if self.xvar[i].vardist == 'frechet':
                        cv = cvi
                    else:
                        cv = cvj
                    f = 1.056 - 0.060 * ro + 0.263 * cv + 0.020 * ro ** 2 + 0.383 * cv ** 2 - 0.332 * ro * cv

                # 15 Xi = gumbel and Xj = weibull - min

                elif self.xvar[i].vardist == 'gumbel' and self.xvar[j].vardist == 'weibull' \
                        or self.xvar[i].vardist == 'weibull' and self.xvar[j].vardist == 'gumbel':
                    if self.xvar[i].vardist == 'weibull':
                        cv = cvi
                    else:
                        cv = cvj
                    f = 1.064 + 0.065 * ro - 0.210 * cv + 0.003 * ro ** 2 + 0.356 * cv ** 2 - 0.211 * ro * cv

                # Table 8 both Xi and Xj belong to group 2: f depends on ro, cvi e cvj

                # 16 Xi = lognorm and Xj = lognorm

                elif self.xvar[i].vardist == 'lognorm' and self.xvar[j].vardist == 'lognorm':
                    f = np.log(1.00 + ro * cvi * cvj)/(ro * np.sqrt(np.log(1.00 + cvi ** 2) * np.log(1.00 + cvj ** 2)))
    

                # 17 Xi = lognorm and Xj = frechet

                elif self.xvar[i].vardist == 'lognorm' and self.xvar[j].vardist == 'frechet' \
                        or self.xvar[i].vardist == 'frechet' and self.xvar[j].vardist == 'lognorm':
                    if self.xvar[i].vardist == 'frechet':
                        cvf = cvi
                        cvl = cvj
                    else:
                        cvf = cvj
                        cvl = cvi
                    f = 1.026 + 0.082 * ro - 0.019 * cvl + 0.222 * cvf \
                        + 0.018 * ro ** 2 + 0.288 * cvl ** 2 + 0.379 * cvf ** 2 \
                        - 0.441 * ro * cvl + 0.126 * cvl * cvf - 0.277 * ro * cvf

                # 18 Xi = lognorm and Xj = weibull - min

                elif self.xvar[i].vardist == 'lognorm' and self.xvar[j].vardist == 'weibull' \
                        or self.xvar[i].vardist == 'weibull' and self.xvar[j].vardist == 'lognorm':
                    if self.xvar[i].vardist == 'weibull':
                        cvw = cvi
                        cvl = cvj
                    else:
                        cvw = cvj
                        cvl = cvi
                    f = 1.031 + 0.052 * ro + 0.011 * cvl - 0.210 * cvw \
                        + 0.002 * ro ** 2 + 0.220 * cvl ** 2 + 0.350 * cvw ** 2 \
                        + 0.005 * ro * cvl + 0.009 * cvl * cvw - 0.174 * ro * cvw

                # 19 Xi = frechet and Xj = frechet

                elif self.xvar[i].vardist == 'frechet' and self.xvar[j].vardist == 'frechet':
                    f = 1.086 + 0.054 * ro + 0.104 * (cvi + cvj) \
                        - 0.055 * ro ** 2 + 0.662 * (cvi ** 2 + cvj ** 2)  \
                        - 0.570 * ro * (cvi + cvj) + 0.203 * cvi * cvj \
                        - 0.020 * ro ** 3 - 0.218 * (cvi ** 3 + cvj ** 3) \
                        - 0.371 * ro * (cvi ** 2 + cvj ** 2) + 0.257 * ro ** 2 * (cvi + cvj) \
                        + 0.141 * cvi * cvj * (cvi + cvj)

                # 20 Xi = frechet and Xj = weibull min

                elif self.xvar[i].vardist == 'frechet' and self.xvar[j].vardist == 'weibull' \
                        or self.xvar[i].vardist == 'weibull' and self.xvar[j].vardist == 'frechet':
                    if self.xvar[i].vardist == 'frechet':
                        cvf = cvi
                        cvw = cvj
                    else:
                        cvf = cvj
                        cvw = cvi
                    f = 1.065 + 0.146 * ro + 0.241 * cvf - 0.259 * cvw \
                        + 0.013 * ro ** 2 + 0.372 * cvf ** 2 + 0.435 * cvw ** 2  \
                        + 0.005 * ro * cvf + 0.034 * cvf * cvw - 0.481 * ro * cvw

                # 20 Xi = weibull and Xj = weibull min

                elif self.xvar[i].vardist == 'weibull' and self.xvar[j].vardist == 'weibull':
                    f = 1.063 - 0.004 * ro - 0.200 * (cvi + cvj) \
                        - 0.001 * ro ** 2 + 0.337 * (cvi ** 2 + cvj ** 2)  \
                        + 0.007 * ro * (cvi + cvj) - 0.007 * cvi * cvj

                # Application of the correction factor f on the ro coefficient
                ro = f * ro
                Rz1[i, j] = ro
                Rz1[j, i] = ro
        return Rz1
  

 