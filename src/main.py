import numpy as np
from scipy.stats import norm
from distribution.map import createDistribution
from utils.validate.domain_types.validate_dvar import ValidateDvar
from utils.validate.domain_types.validate_gx import ValidateGx
from utils.validate.domain_types.validate_corrmatrix import ValidateCorrelationMatrix
from correlation import Correlation
from methods.transformation import TransformationMethods
from methods.simulation.monte_carlo import MonteCarloMethods


class Reliability():

    def __init__(self, xvar, dvar, gx, x0=None, Rz=None):
        
        #Simple setters
        self.dvar = dvar 
        self.fel = gx
        self.ndvar = len(dvar)
        self.nxvar = len(xvar)

        #Data processing - Complex setters
        self._xvar = self.set_xvar(xvar, x0)
        self._Rz = self.set_Rz(Rz)
        self._d = self.set_d(self.dvar, self.ndvar)
        self._x0 = self.set_x0(x0, self.nxvar, self.xvar)

        self._correlation = Correlation({'Rz': self._Rz, 'xvar': self._xvar, 'nxvar': self.nxvar})
        self._transformation =  TransformationMethods(self)
        self._simulation = MonteCarloMethods(self)
           
    @property
    def xvar(self):
      return self._xvar

    @property
    def dvar(self):
      return self._dvar

    @property
    def gx(self):
      return self._gx

    @property
    def Rz(self):
      return self._Rz

    @property
    def x0(self):
      return self._x0

    @property
    def nxvar(self):
      return len(self.xvar)
    
    @property
    def ndvar(self):
      return len(self.dvar)
    
    @property
    def d(self):
        return self._d
    
    @property
    def correlation(self):
      return self._correlation

    @xvar.setter
    def xvar(self, xvar):
        self._xvar = xvar

    @dvar.setter
    def dvar(self, dvar): 
      ValidateDvar(dvar)
      self._dvar = dvar

    @gx.setter
    def gx(self, gx): 
      ValidateGx(gx, self.nxvar, self.ndvar)
      self._gx = gx

    @Rz.setter
    def Rz(self, Rz):
        self._Rz = Rz

    @x0.setter
    def x0(self, x0):
        self._x0 = x0 

    def set_xvar(self, xvar, x0): 
      nxvar = len(xvar)
      for i, var in enumerate(xvar):              
          xvar[i] = createDistribution(var)

      if x0 is None:
          x0 = np.zeros(nxvar)
          for i, var in enumerate(xvar):
              x0[i] = float(var.varmean)
              var.varhmean = float(var.varmean)
      else:
          for i, var in enumerate(xvar):
              var.varhmean = x0[i]

      return xvar

    def set_Rz(self, Rz):
      if Rz is None:
          return np.eye(self.nxvar)
      else:
        ValidateCorrelationMatrix(Rz)
        return np.array(Rz)
        
    @nxvar.setter
    def nxvar(self, nxvar): 
      self._nxvar = nxvar
  
    @ndvar.setter
    def ndvar(self, ndvar): 
      self._ndvar = ndvar

    @correlation.setter
    def correlation(self, correlation): 
      self._correlation = correlation

    def set_d(self, dvar, ndvar):
      #
      # Initial values of the design variables
      #
      i = -1
      d = np.zeros(ndvar)
      for var in dvar:
          i += 1
          # Mean value of the random variables x
          d[i] = float(var['varvalue'])
      
      #
      # Setting variables initial values
      #
      return d

    def set_x0(self, x0, nxvar, xvar):
      if x0 is None:
          # Original mean of the variables x
          x0 = np.zeros(nxvar)
          for i, var in enumerate(xvar):
              # Mean value of the random variables x
              x0[i] = float(var.varmean)
              var.varhmean = float(var.varmean)
      else:
          for i, var in enumerate(xvar):
              var.varhmean = x0[i]

      return x0

    def __getattr__(self, name):
      for module in [self._transformation, self._simulation]:
        if hasattr(module, name):
          return getattr(module, name)
      raise AttributeError(f"'Reliability' object has no attribute '{name}'")

    def multig(self, xvar, dvar, glist, iprint=True):
        """
        Solution of the problem of the reliability of serial system with multiple limit state functions
        According to:
        BECK, A.T.
        Confiabilidade e Segurança das Estruturas
        Rio de Janeiro, Elsevier, 2019.
        """
        ng = int(len(glist))
        nvar = int(len(xvar))
        beta = np.zeros(ng)
        alpha = np.zeros((ng, nvar))
        pf = np.zeros(ng)
        pa = np.zeros((ng, ng))
        pb = np.zeros((ng, ng))
        pfij_inf = np.zeros((ng, ng))
        pfij_sup = np.zeros((ng, ng))

        i = -1
        for gfunction in glist:
            i += 1
            #
            # FORM method for  the multiple g(x) functions
            #
            test = Reliability(xvar, dvar, gfunction, None, None)
            beta[i], x0, alpha[i, :], normgradyk, sigmaxneqk = test.form(iHLRF=True, toler=1.e-3)
            pf[i] = norm.cdf(-beta[i])
        #

        pfinf = pf.max()
        pfsup = pf.sum()
        alpha_sign = np.sign(alpha)
        alpha2 = alpha_sign * alpha ** 2

        #
        # Sort arrays pf, beta, alpha, alpha2, glist in decrescent order of probability of failure
        #
        glist = np.array(glist)
        ig = (-pf).argsort()
        pf = pf[ig]
        beta = beta[ig]
        alpha = alpha[ig, :]
        alpha2 = alpha2[ig, :]
        glist = glist[ig]


        #
        # Calculation of the correlation coefficients between the limit state functions
        #

        ro = np.dot(alpha, alpha.T)

        #
        # Calculation of the p(Aij) and p(Bij) matrices
        #

        for i in range(ng):
            for j in range(ng):
                if i != j:
                    pa[i, j] = norm.cdf(-beta[i]) * norm.cdf(
                        -((beta[j] - ro[i, j] * beta[i]) / np.sqrt(1. - ro[i, j] ** 2)))
                    pb[i, j] = norm.cdf(-beta[j]) * norm.cdf(
                        -((beta[i] - ro[i, j] * beta[j]) / np.sqrt(1. - ro[i, j] ** 2)))


        #
        # Calculation of the terms P(Fi.Fj)sup and P(Fi.Fj)inf
        #

        for i in range(ng):
            for j in range(ng):
                if i != j:
                    if ro[i, j] >= 0.00:
                        pfij_inf[i, j] = pa[i, j] + pb[i, j]
                        pfij_sup[i, j] = np.max([pa[i, j], pb[i, j]])
                    else:
                        pfij_inf[i, j] = np.min([pa[i, j], pb[i, j]])
                        pfij_sup[i, j] = 0.00

        #
        # Calculation of inferior and superior limits for the probability of failure of the system
        #
        #
        # Inferior limit: pf_inf
        #
        pf_inf = pf[0]
        for i in range(1, ng, 1):
            pfi_inf = pf[i]
            for j in range(0, i, 1):
                pfi_inf -= pfij_inf[i, j]
            pf_inf += np.max([0, pfi_inf])

        #
        # Superior limit: pf_sup
        #
        pf_sup = sum(pf)
        for i in range(1, ng, 1):
            pf_sup -= np.max(pfij_sup[i, 0:i], axis=0)

        beta_sup = -norm.ppf(pf_inf)
        beta_inf = -norm.ppf(pf_sup)
        glist = list(glist)

        if iprint:
            #
            # Print the initial results
            #
            print('Initial results:')
            print('pf =', pf)
            print('beta =', beta)
            print('pfinf =', pfinf)
            print('pfsup =', pfsup)
            print('alpha =', alpha)
            print(('alpha2 =', alpha2))
            print('ro =', ro)
            print('pa =', pa)
            print('pb =', pb)
            #
            # Print final results
            #
            print('Final results:')
            print('g list =', glist)
            print('pf =', pf)
            print('beta =', beta)
            print('alpha =', alpha)
            print(('alpha2 =', alpha2))
            print('pf_inf =', pf_inf)
            print('pf_sup =', pf_sup)
            print('beta_inf =', beta_inf)
            print('beta_sup =', beta_sup)

