import numpy as np
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import uniform
from scipy.stats import gumbel_r
from scipy.stats import invweibull
from scipy.stats import weibull_min
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
import scipy.optimize
from scipy import optimize
import scipy.linalg
from scipy.special import gamma
import pandas as pd
from visualize import DataVisualizeTransformation

class TransformationBase:

  def mfunc(self, normy, g, c):
    # Penalty function m(y) for FORM-iHLRF algorithm
    my = 1. / 2. * normy ** 2 + c * np.abs(g)
    return my
  
  def fkapa(self, kapa, deltax, gsignal):
    # Evaluation of parameter k for Frechet and Weibull distribution min
    fk = 1.00 + deltax ** 2 - gamma(1.00 + gsignal * 2.00 / kapa) / gamma(1.00 + gsignal * 1.00 / kapa) ** 2
    return fk
  
class TransformationMethods(TransformationBase):
    
    def __init__(self, parent):
      ## Get all properties about main class: Reliability
      self.reliability = parent
      
    def form(self, iHLRF, toler=1.e-3, iprint=True):
      """

              Algorithm FORM-iHLRF. Normal equivalente transformation

      """
    #
      # FORM - First Order Reliability Method with improved HLRF (iHLRF)
      #
      #
      # Equivalent normal distribution parameters
      # xval = value of the variable x (scalar)
      # xpar1,xpar2,xpar3,xpar4 = parameters of the original pdf (scalars)
      # namedist = name of the x probability distribution ('string')
      #

      def normeqv(xval, xpar1, xpar2, xpar3, xpar4, namedist):

          #
          # Normal distribution
          #
          if namedist.lower() == 'gauss':
              mux = xpar1
              sigmax = xpar2
              muxneq = mux
              sigmaxneq = sigmax
          #
          # Uniform or constant distribution
          #
          elif namedist.lower() == 'uniform':
              a = xpar1
              b = xpar2
              c = (b - a)
              pdfx = 1. / c
              cdfx = (xval - a) / c
              zval = norm.ppf(cdfx)
              sigmaxneq = (norm.pdf(zval)) / pdfx
              muxneq = xval - zval * sigmaxneq
          #
          # Lognormal distribution
          #
          elif namedist.lower() == 'lognorm':
              mux = xpar1
              sigmax = xpar2
              zetax = np.sqrt(np.log(1. + (sigmax / mux) ** 2))
              lambdax = np.log(mux) - 0.50 * zetax ** 2
              sigmaxneq = zetax * xval
              muxneq = xval * (1. - np.log(xval) + lambdax)
          #
          # Gumbel distribution
          #
          elif namedist.lower() == 'gumbel':
              mux = xpar1
              sigmax = xpar2
              alphan = (np.pi / np.sqrt(6.00)) / (sigmax)
              un = mux - np.euler_gamma / alphan
              cdfx = np.exp(-np.exp(-alphan * (xval - un)))
              pdfx = alphan * np.exp(-alphan * (xval - un)) * cdfx
              zval = norm.ppf(cdfx)
              sigmaxneq = norm.pdf(zval) / pdfx
              muxneq = xval - zval * sigmaxneq
          #
          #
          # Frechet distribution
          #
          elif namedist.lower() == 'frechet':
              mux = xpar1
              sigmax = xpar2
              deltax = sigmax / mux
              kapa0 = 2.50
              gsignal = -1.00
              kapa = scipy.optimize.newton(self.fkapa, kapa0, args=(deltax, gsignal))
              vn = mux / gamma(1.00 - 1.00 / kapa)
              cdfx = np.exp(-(vn / xval) ** kapa)
              pdfx = kapa / vn * (vn / xval) ** (kapa + 1) * np.exp(-(vn / xval) ** kapa)
              zval = norm.ppf(cdfx)
              sigmaxneq = norm.pdf(zval) / pdfx
              muxneq = xval - zval * sigmaxneq
          #
          #
          # Weibull distribution minimum
          #
          elif namedist.lower() == 'weibull':
              mux = xpar1
              sigmax = xpar2
              epsilon = xpar3
              deltax = sigmax / (mux - epsilon)
              kapa0 = 2.50
              gsignal = 1.00
              kapa = scipy.optimize.newton(self.fkapa, kapa0, args=(deltax, gsignal))
              w1 = (mux - epsilon) / gamma(1.00 + 1.00 / kapa) + epsilon
              y1 = (xval - epsilon) / (w1 - epsilon)
              pdfx = weibull_min.pdf(y1, kapa) / (w1 - epsilon)
              cdfx = weibull_min.cdf(y1, kapa)
              zval = norm.ppf(cdfx)
              sigmaxneq = norm.pdf(zval) / pdfx
              muxneq = xval - zval * sigmaxneq
          #
          #
          # Beta distribution
          #
          elif namedist.lower() == 'beta':
              a = xpar1
              b = xpar2
              q = xpar3
              r = xpar4
              loc = a
              scale = (b - a)
              pdfx = beta_dist.pdf(xval, q, r, loc, scale)
              cdfx = beta_dist.cdf(xval, q, r, loc, scale)
              zval = norm.ppf(cdfx)
              sigmaxneq = norm.pdf(zval) / pdfx
              muxneq = xval - zval * sigmaxneq


          #
          #
          # Gamma distribution
          #
          elif namedist.lower() == 'gamma':
              mux = xpar1
              sigmax = xpar2
              delta = sigmax / mux
              k = 1. / delta ** 2
              v = k / mux
              a = k
              loc = 0.00
              scale = 1. / v
              pdfx = gamma_dist.pdf(xval, a, loc, scale)
              cdfx = gamma_dist.cdf(xval, a, loc, scale)
              zval = norm.ppf(cdfx)
              sigmaxneq = norm.pdf(zval) / pdfx
              muxneq = xval - zval * sigmaxneq

          return muxneq, sigmaxneq
      
      #
      #
      # Data input
      #
      # Number of variables of the problem

      # Equivalent normal mean and standard deviation of the variables
      muxneqk = np.zeros(self.reliability.nxvar)
      sigmaxneqk = np.zeros(self.reliability.nxvar)
      namevar = []
      dist = []
      mux0 = []
      sigmax0 = []
      par1 = []
      par2 = []
      par3 = []
      par4 = []
      #
      # Original mean and standard deviation of the variables x
      #

      i = -1
      for var in self.reliability.xvar:
        i += 1

        # Names of the random variables x
        namevar.append(str(var.varname))

        # Names of the probability density functions of the variables x
        dist.append(str(var.vardist))

        # Mean value of the random variables x
        mux0.append(float(var.varmean))

        # Standard deviation of the random variables x
        if var.varstd == 0.00:
            sigmax0.append(float(var.varcov) * float(var.varmean))
        else:
            sigmax0.append(float(var.varstd))

        # Parameter1
        if hasattr(var, 'parameter1'):
            par1.append(float(var.parameter1))
        else:
            par1.append(0.00)

        # Parameter2
        if hasattr(var, 'parameter2'):
            par2.append(float(var.parameter2))
        else:
            par2.append(0.00)

        # Parameter3
        if hasattr(var, 'parameter3'):
            par3.append(float(var.parameter3))
        else:
            par3.append(0.00)

        # Parameter4
        if hasattr(var, 'parameter4'):
            par4.append(float(var.parameter4))
        else:
            par4.append(0.00)
          
          
      #
      # Conversion to array format
      #
      mux0 = np.array(mux0)
      sigmax0 = np.array(sigmax0)
      par1 = np.array(par1)
      par2 = np.array(par2)
      par3 = np.array(par3)
      par4 = np.array(par4)
      #
      #   Algorithm FORM-HLRF: Beck, 2019, pag. 101.
      #
      #
      # Step 1 - Determination of equivalent correlation coefficients and
      #          Jacobian matrices Jxz and Jzx
      #
      Imatrix = np.eye(self.reliability.nxvar)
      #
      # Correlation matrix is self.corrmatrix
      #
      if iprint:
          print('Correlation Matrix after Nataf correction:')
          print(self.reliability.correlation.Rz_rectify)
      #
      # Cholesky decomposition of the correlation matrix
      #
      L = scipy.linalg.cholesky(self.reliability.correlation.Rz_rectify, lower=True)
      Jzy = np.copy(L)
      Jyz = np.linalg.inv(L)
      #
      # Step 2 - Initialize de xk value with mux0
      #
      # Initialization of the variable yk1
      # Jacobian matrices of x==>z and z==>y transformations
      D = sigmax0 * Imatrix
      Jzx = np.linalg.inv(D)
      Jyx = np.dot(Jyz, Jzx)
      Jxz = np.copy(D)
      Jxy = np.dot(Jxz, Jzy)
      yk1 = np.zeros(self.reliability.nxvar)
  #    xk1 = mux0 + Jxy.dot(yk1)
      xk1 = np.copy(self.reliability.x0)
      #
      # Error tolerance for yk and g(x)
      epsilon = toler
      delta = toler * np.abs(self.reliability.fel(xk1, self.reliability.d))
      # Initial values for errors and iteration counters
      erro1 = 1000.00
      erro2 = 1000.00
      kiter = 0
      # Value of dx increment for the evaluation of the derivatives
      eps = 1.e-6
      #
      while (erro1 > epsilon or erro2 > delta) and kiter < 100:
          #
          kiter += 1
          xk = np.copy(xk1)
          #
          # Calculation of the equivalent normal distribution parameters for xk
          #
          for i in range(self.reliability.nxvar):
              xval = xk[i]
              mux = mux0[i]
              sigmax = sigmax0[i]
              namedist = dist[i]
              xpar1 = mux
              xpar2 = sigmax
              xpar3 = par3[i]
              xpar4 = par4[i]
              if dist[i] == 'beta':
                  xpar1 = par1[i]
                  xpar2 = par2[i]
              if dist[i] == 'uniform':
                  xpar1 = par1[i]
                  xpar2 = par2[i]

              muxneqk[i], sigmaxneqk[i] = normeqv(xval, xpar1, xpar2, xpar3, xpar4, namedist)
          #
          # Step 3 - Update of the Jacobian matrices Jyx and Jxy
          #
          Dneq = sigmaxneqk * Imatrix
          Jzx = np.linalg.inv(Dneq)
          Jyx = np.dot(Jyz, Jzx)
          Jxz = np.copy(Dneq)
          Jxy = np.dot(Jxz, Jzy)
          #
          #  Step 4 - Transformation from xk to yk
          #
          yk = Jyx.dot(xk - muxneqk)
          normyk = np.linalg.norm(yk)
          beta = np.linalg.norm(yk)

          #
          #  Step 5 - Evaluation of g(xk)
          #
          gxk = self.reliability.fel(xk, self.reliability.d)

          #
          # Step 6 - Evaluation of the gradients of g(x) in relation to yk
          #
          #
          # a. Calculation of the partial derivatives of g(x) in relation to xk
          #
          gradxk = optimize.approx_fprime(xk, self.reliability.fel, eps, self.reliability.d)
          #
          # b. Calculation of the partial derivatives of g(x) in relation to yk
          #
          gradyk = np.transpose(Jxy).dot(gradxk)
          normgradyk = np.linalg.norm(gradyk)
          #
          # c. Calculation of the direction cosines for xk
          #
          # Direction cosines
          alpha = gradyk / normgradyk

          #
          # Step 7. Vector yk updating to yk+1 by HLRF algorithm
          #
          dk = ((np.dot(gradyk, yk) - gxk) / normgradyk ** 2) * gradyk - yk
          lambdak = 1.00
          yk1 = yk + lambdak * dk
          #
          # Parameters of iHLRF method
          #
          if iHLRF:
              gamma0 = 2.0
              a = 0.1
              # a = 0.5
              b = 0.5
              #
              gyk = gxk
              normyk = np.linalg.norm(yk)
              normyk1 = np.linalg.norm(yk1)
              c1 = normyk / normgradyk
              #
              if erro2 > delta:
                  c2 = 0.5 * normyk1 ** 2 / np.abs(gyk)
                  ck = gamma0 * np.max([c1, c2]) 
              else:
                  ck = gamma0 * c1
              #
              k = -1
              f1 = 1.00
              f2 = 0.00
              while f1 > f2 and k < 10:
                  k += 1
                  lambdak = b ** k
                  yk1 = yk + lambdak * dk
                  xk1 = muxneqk + Jxy.dot(yk1)
                  gyk1 = self.reliability.fel(xk1, self.reliability.d)
                  normyk1 = np.linalg.norm(yk1)
                  f1 = self.mfunc(normyk1, gyk1, ck) - self.mfunc(normyk, gyk, ck)
                  gradm = yk + ck * gradyk * np.sign(gyk)
                  normgradm = np.linalg.norm(gradm)
                  f2 = a * lambdak * np.dot(gradm, dk)
          #        f2=-a*lambdak*normgradm**2 # Beck pg. 85: It does not work!!
          #        res=np.array([k,ck,lambdak,gxk,gyk1,f1,f2])
          #        print(res)
          #
          yk1 = yk + lambdak * dk

          #
          # Step 8. Transformation from yk+1 to xk+1
          #
          xk1 = muxneqk + Jxy.dot(yk1)

          #
          # Step 9. Convergence test for yk and g(x)
          #
          prod = normgradyk * normyk
          # Evaluation of the error in the yk1 vector
          if np.abs(prod) > eps:
              erro1 = 1. - np.abs(np.dot(gradyk, yk) / (normgradyk * normyk))
          else:
              erro1 = 1000.00
          # Evaluation of the error in the limit state function g(x)
          erro2 = np.abs(gxk)
          # Printing of the updated values
          if iprint:
            DataVisualizeTransformation.one_cycle_print_results({
              'kiter': kiter,
              'gxk': gxk,
              'erro1': erro1,
              'beta': beta,
              'namevar': namevar,
              'dist': dist,
              'muxneqk': muxneqk,
              'sigmaxneqk': sigmaxneqk,
              'xk': xk,
              'yk': yk,
              'alpha': alpha
            })
      #
      pf = norm.cdf(-beta)
      if iprint: DataVisualizeTransformation.print_results({'pf': pf, 'beta': beta})

      return {
        "beta": beta,
        "yk":yk,
        "xk": xk,
        "alpha": alpha,
        "normgradyk": normgradyk,
        "sigmaxneqk": sigmaxneqk
        } 

    def form2(self, iHLRF, toler=1.e-3, iprint=True):
        """

            Algorithm FORM-iHLRF. Direct mapping to standard Gaussian space

        """
      #
        # FORM - First Order Reliability Method with improved HLRF (iHLRF)

        #
        # Equivalent normal distribution parameters
        # xval = value of the variable x (scalar)
        # xpar1,xpar2,xpar3,xpar4 = parameters of the original pdf (scalars)
        # namedist = name of the x probability distribution ('string')
        # zval = equivalente normal variabel correlated
        #
        def normeqv(xval, xpar1, xpar2, xpar3, xpar4, namedist):

            #
            # Normal distribution
            #
            if namedist.lower() == 'gauss':
                mux = xpar1
                sigmax = xpar2
                muxneq = mux
                sigmaxneq = sigmax
            #
            # Uniform or constant distribution
            #
            elif namedist.lower() == 'uniform':
                epsilon = 1.e-8
                a = xpar1
                b = xpar2
                c = (b - a)
                if xval<=a: xval = a + epsilon
                if xval>=b: xval = b - epsilon
                pdfx = 1. / c
                cdfx = (xval - a) / c
                zval = norm.ppf(cdfx)
                sigmaxneq = (norm.pdf(zval)) / pdfx
                muxneq = xval - zval * sigmaxneq
            #
            # Lognormal distribution
            #
            elif namedist.lower() == 'lognorm':
                mux = xpar1
                sigmax = xpar2
                zetax = np.sqrt(np.log(1. + (sigmax / mux) ** 2))
                lambdax = np.log(mux) - 0.50 * zetax ** 2
                sigmaxneq = zetax * xval
                muxneq = xval * (1. - np.log(xval) + lambdax)
            #
            # Gumbel distribution
            #
            elif namedist.lower() == 'gumbel':
                mux = xpar1
                sigmax = xpar2
                alphan = (np.pi / np.sqrt(6.00)) / (sigmax)
                un = mux - np.euler_gamma / alphan
                cdfx = np.exp(-np.exp(-alphan * (xval - un)))
                pdfx = alphan * np.exp(-alphan * (xval - un)) * cdfx
                zval = norm.ppf(cdfx)
                sigmaxneq = norm.pdf(zval) / pdfx
                muxneq = xval - zval * sigmaxneq
            #
            #
            # Frechet distribution
            #
            elif namedist.lower() == 'frechet':
                mux = xpar1
                sigmax = xpar2
                deltax = sigmax / mux
                kapa0 = 2.50
                gsignal = -1.00
                kapa = scipy.optimize.newton(self.fkapa, kapa0, args=(deltax, gsignal))
                vn = mux / gamma(1.00 - 1.00 / kapa)
                cdfx = np.exp(-(vn / xval) ** kapa)
                pdfx = kapa / vn * (vn / xval) ** (kapa + 1) * np.exp(-(vn / xval) ** kapa)
                zval = norm.ppf(cdfx)
                sigmaxneq = norm.pdf(zval) / pdfx
                muxneq = xval - zval * sigmaxneq
            #
            #
            # Weibull distribution minimum
            #
            elif namedist.lower() == 'weibull':
                mux = xpar1
                sigmax = xpar2
                epsilon = xpar3
                deltax = sigmax / (mux - epsilon)
                kapa0 = 2.50
                gsignal = 1.00
                kapa = scipy.optimize.newton(self.fkapa, kapa0, args=(deltax, gsignal))
                w1 = (mux - epsilon) / gamma(1.00 + 1.00 / kapa) + epsilon
                y1 = (xval - epsilon) / (w1 - epsilon)
                pdfx = weibull_min.pdf(y1, kapa) / (w1 - epsilon)
                cdfx = weibull_min.cdf(y1, kapa)
                zval = norm.ppf(cdfx)
                sigmaxneq = norm.pdf(zval) / pdfx
                muxneq = xval - zval * sigmaxneq
            #
            #
            # Beta distribution
            #
            elif namedist.lower() == 'beta':
                epsilon = 1.e-8
                a = xpar1
                b = xpar2
                q = xpar3
                r = xpar4
                loc = a
                scale = (b - a)
                if xval<=a: xval = a + epsilon
                if xval>=b: xval = b - epsilon
                pdfx = beta_dist.pdf(xval, q, r, loc, scale)
                cdfx = beta_dist.cdf(xval, q, r, loc, scale)
                zval = norm.ppf(cdfx)
                sigmaxneq = norm.pdf(zval) / pdfx
                muxneq = xval - zval * sigmaxneq

            #
            #
            # Gamma distribution
            #
            elif namedist.lower() == 'gamma':
                mux = xpar1
                sigmax = xpar2
                deltax = sigmax / mux
                k = 1. / deltax
                v = k / mux
                a = k
                loc = 0.00
                scale = 1. / v
                pdfx = gamma_dist.pdf(xval, a, loc, scale)
                cdfx = gamma_dist.cdf(xval, a, loc, scale)
                zval = norm.ppf(cdfx)
                sigmaxneq = norm.pdf(zval) / pdfx
                muxneq = xval - zval * sigmaxneq

            return muxneq, sigmaxneq
        
        def direct_mapping(xval, xpar1, xpar2, xpar3, xpar4, namedist):

            #
            # Normal distribution. Direct mapping to standard Gaussian space
            #
            if namedist.lower() == 'gauss':
                mux = xpar1
                sigmax = xpar2
                cdfx = norm.cdf(xval,mux,sigmax)
                
            #
            # Uniform or constant distribution. Direct mapping to standard Gaussian space
            #
            elif namedist.lower() == 'uniform':
                epsilon = 1.e-8
                a = xpar1
                b = xpar2
                c = (b - a)
                if xval<=a: xval = a + epsilon
                if xval>=b: xval = b - epsilon
                cdfx = uniform.cdf(xval,a,c)
                
                
            #
            # Lognormal distribution. Direct mapping to standard Gaussian space
            #
            elif namedist.lower() == 'lognorm':
                mux = xpar1
                sigmax = xpar2
                zetax = np.sqrt(np.log(1. + (sigmax / mux) ** 2))
                lambdax = np.log(mux) - 0.50 * zetax ** 2
                cdfx = lognorm.cdf(xval/np.exp(lambdax), zetax)
                
            #
            # Gumbel distribution. Direct mapping to standard Gaussian space
            #
            elif namedist.lower() == 'gumbel':
                mux = xpar1
                sigmax = xpar2
                alphan = (np.pi / np.sqrt(6.00)) / (sigmax)
                un = mux - np.euler_gamma / alphan
                betan = 1.00 / alphan
                cdfx = gumbel_r.cdf(xval, un, betan)
                
            #
            #
            # Frechet distribution. Direct mapping to standard Gaussian space
            #
            elif namedist.lower() == 'frechet':
                mux = xpar1
                sigmax = xpar2
                deltax = sigmax / mux
                kapa0 = 2.50
                gsignal = -1.00
                kapa = scipy.optimize.newton(self.fkapa, kapa0, args=(deltax, gsignal))
                vn = mux / gamma(1.00 - 1.00 / kapa)
                yn = xval / vn
                cdfx = invweibull.cdf(yn, kapa)
                
            #
            #
            # Weibull distribution minimum. Direct mapping to standard Gaussian space
            #
            elif namedist.lower() == 'weibull':
                mux = xpar1
                sigmax = xpar2
                epsilon = xpar3
                deltax = sigmax / (mux - epsilon)
                kapa0 = 2.50
                gsignal = 1.00
                kapa = scipy.optimize.newton(self.fkapa, kapa0, args=(deltax, gsignal))
                w1 = (mux - epsilon) / gamma(1.00 + 1.00 / kapa) + epsilon
                yn = (xval - epsilon) / (w1 - epsilon)
                cdfx = weibull_min.cdf(yn, kapa)
                
            #
            #
            # Beta distribution. Direct mapping to standard Gaussian space
            #
            elif namedist.lower() == 'beta':
                epsilon = 1.e-8
                a = xpar1
                b = xpar2
                q = xpar3
                r = xpar4
                loc = a
                scale = (b - a)
                if xval<=a: xval = a + epsilon
                if xval>=b: xval = b - epsilon
                cdfx = beta_dist.cdf(xval, q, r, loc, scale)

            #
            #
            # Gamma distribution. Direct mapping to standard Gaussian space
            #
            elif namedist.lower() == 'gamma':
                mux = xpar1
                sigmax = xpar2
                delta = sigmax / mux
                k = 1. / delta ** 2
                v = k / mux
                a = k
                loc = 0.00
                scale = 1. / v
                cdfx = gamma_dist.cdf(xval, a, loc, scale)
                

            return cdfx
        
        
        def inverse_mapping(zval, xpar1, xpar2, xpar3, xpar4, namedist):

            #
            # Normal distribution. Inverse mapping from standard Gaussian space
            #
            if namedist.lower() == 'gauss':
                mux = xpar1
                sigmax = xpar2
                xval = norm.ppf(cdfx,mux,sigmax)
            #
            # Uniform or constant distribution. Inverse mapping from standard Gaussian space
            #
            elif namedist.lower() == 'uniform':
                a = xpar1
                b = xpar2
                c = (b - a)
                xval = uniform.ppf(cdfx,a,c)
            #
            # Lognormal distribution. Inverse mapping from standard Gaussian space
            #
            elif namedist.lower() == 'lognorm':
                mux = xpar1
                sigmax = xpar2
                zetax = np.sqrt(np.log(1. + (sigmax / mux) ** 2))
                lambdax = np.log(mux) - 0.50 * zetax ** 2
                xval = lognorm.ppf(cdfx, zetax)*np.exp(lambdax)
            #
            # Gumbel distribution. Inverse mapping from standard Gaussian space
            #
            elif namedist.lower() == 'gumbel':
                mux = xpar1
                sigmax = xpar2
                alphan = (np.pi / np.sqrt(6.00)) / (sigmax)
                un = mux - np.euler_gamma / alphan
                xval = un - 1./alphan * np.log(np.log(1. / cdfx))
            #
            #
            # Frechet distribution. Inverse mapping from standard Gaussian space
            #
            elif namedist.lower() == 'frechet':
                mux = xpar1
                sigmax = xpar2
                deltax = sigmax / mux
                kapa0 = 2.50
                gsignal = -1.00
                kapa = scipy.optimize.newton(self.fkapa, kapa0, args=(deltax, gsignal))
                vn = mux / gamma(1.00 - 1.00 / kapa)
                xval = vn / (np.log(1. / cdfx)) ** (1. / kapa)
            #
            #
            # Weibull distribution minimum. Inverse mapping from standard Gaussian space
            #
            elif namedist.lower() == 'weibull':
                mux = xpar1
                sigmax = xpar2
                epsilon = xpar3
                deltax = sigmax / (mux - epsilon)
                kapa0 = 2.50
                gsignal = 1.00
                kapa = scipy.optimize.newton(self.fkapa, kapa0, args=(deltax, gsignal))
                w1 = (mux - epsilon) / gamma(1.00 + 1.00 / kapa) + epsilon
                xval = (w1 - epsilon) * (np.log(1./(1. - cdfx))) ** (1. / kapa) + epsilon
            #
            #
            # Beta distribution. Inverse mapping from standard Gaussian space
            #
            elif namedist.lower() == 'beta':
                a = xpar1
                b = xpar2
                q = xpar3
                r = xpar4
                loc = a
                scale = (b - a)
                xval = beta_dist.ppf(cdfx, q, r, loc, scale)

            #
            #
            # Gamma distribution. Inverse mapping from standard Gaussian space
            #
            elif namedist.lower() == 'gamma':
                mux = xpar1
                sigmax = xpar2
                delta = sigmax / mux
                k = 1. / delta ** 2
                v = k / mux
                a = k
                loc = 0.00
                scale = 1. / v
                xval = gamma_dist.ppf(cdfx, a, loc, scale)
                

            return xval
        #
        #
        # Data input
        #
        # Number of variables of the problem

        # Equivalent normal mean and standard deviation of the variables
        muxneqk = np.zeros(self.reliability.nxvar)
        sigmaxneqk = np.zeros(self.reliability.nxvar)
        namevar = []
        dist = []
        mux0 = []
        sigmax0 = []
        par1 = []
        par2 = []
        par3 = []
        par4 = []
        #
        # Original mean and standard deviation of the variables x
        #

        i = -1
        for var in self.reliability.xvar:
          i += 1

          # Names of the random variables x
          namevar.append(str(var.varname))

          # Names of the probability density functions of the variables x
          dist.append(str(var.vardist))

          # Mean value of the random variables x
          mux0.append(float(var.varmean))

          # Standard deviation of the random variables x
          if var.varstd == 0.00:
              sigmax0.append(float(var.varcov) * float(var.varmean))
          else:
              sigmax0.append(float(var.varstd))

          # Parameter 1
          if hasattr(var, 'parameter1'):
              par1.append(float(var.parameter1))
          else:
              par1.append(0.00)

          # Parameter 2
          if hasattr(var, 'parameter2'):
              par2.append(float(var.parameter2))
          else:
              par2.append(0.00)

          # Parameter 3
          if hasattr(var, 'parameter3'):
              par3.append(float(var.parameter3))
          else:
              par3.append(0.00)

          # Parameter 4
          if hasattr(var, 'parameter4'):
              par4.append(float(var.parameter4))
          else:
              par4.append(0.00)
            
           
        #
        # Conversion to array format
        #
        mux0 = np.array(mux0)
        sigmax0 = np.array(sigmax0)
        par1 = np.array(par1)
        par2 = np.array(par2)
        par3 = np.array(par3)
        par4 = np.array(par4)
        #
        #   Algorithm FORM-HLRF: Beck, 2019, pag. 101.
        #
        #
        # Step 1 - Determination of equivalent correlation coefficients and
        #          Jacobian matrices Jxz and Jzx
        #
        Imatrix = np.eye(self.reliability.nxvar)
        #
        # Correlation matrix is self.corrmatrix
        #
        if iprint:
            print('Correlation Matrix after Nataf correction:')
            print(self.reliability.correlation.Rz_rectify)
        #
        # Cholesky decomposition of the correlation matrix
        #
        L = scipy.linalg.cholesky(self.reliability.correlation.Rz_rectify, lower=True)
        Jzy = np.copy(L)
        Jyz = np.linalg.inv(L)
        #
        # Step 2 - Initialize de xk value with mux0
        #
        # Initialization of the variable yk1
        # Jacobian matrices of x==>z and z==>y transformations
        D = sigmax0 * Imatrix
        Jzx = np.linalg.inv(D)
        Jyx = np.dot(Jyz, Jzx)
        Jxz = np.copy(D)
        Jxy = np.dot(Jxz, Jzy)
        yk1 = np.zeros(self.reliability.nxvar)
        zk = np.zeros(self.reliability.nxvar)
        zk1 = np.zeros(self.reliability.nxvar)
        uk = np.zeros(self.reliability.nxvar)
        uk1 = np.zeros(self.reliability.nxvar)
        xk1 = np.copy(self.reliability.x0)
        #
        # Error tolerance for yk and g(x)
        epsilon = toler
        delta = toler * np.abs(self.reliability.fel(xk1, self.reliability.d))
        # Initial values for errors and iteration counters
        erro1 = 1000.00
        erro2 = 1000.00
        kiter = 0
        # Value of dx increment for the evaluation of the derivatives
        eps = 1.e-6
        #
        while (erro1 > epsilon or erro2 > delta) and kiter < 100:
            #
            kiter += 1
            xk = np.copy(xk1)
            #
            # Step 2 - Calculation of equivalent normal variables by direct mapping to standard normal space (correlated)
            #
            for i in range(self.reliability.nxvar):
                xval = xk[i]
                mux = mux0[i]
                sigmax = sigmax0[i]
                namedist = dist[i]
                xpar1 = mux
                xpar2 = sigmax
                xpar3 = par3[i]
                xpar4 = par4[i]
                if dist[i] == 'beta':
                    xpar1 = par1[i]
                    xpar2 = par2[i]
                if dist[i] == 'uniform':
                    xpar1 = par1[i]
                    xpar2 = par2[i]

                uk[i] = direct_mapping(xval, xpar1, xpar2, xpar3, xpar4, namedist)

                muxneqk[i], sigmaxneqk[i] = normeqv(xval, xpar1, xpar2, xpar3, xpar4, namedist)
            #
            # Step 3 - Update of the Jacobian matrices Jyx and Jxy
            #
            Dneq = sigmaxneqk * Imatrix
            Jzx = np.linalg.inv(Dneq)
            Jyx = np.dot(Jyz, Jzx)
            Jxz = np.copy(Dneq)
            Jxy = np.dot(Jxz, Jzy)
            
            
            
            
            #
            #  Step 4 - Transformation from zk to yk
            #
            zk = norm.ppf(uk)
            yk = Jyz.dot(zk)
            normyk = np.linalg.norm(yk)
            beta = np.linalg.norm(yk)

            #
            #  Step 5 - Evaluation of g(xk)
            #
            gxk = self.reliability.fel(xk, self.reliability.d)

            #
            # Step 6 - Evaluation of the gradients of g(x) in relation to yk
            #
            #
            # a. Calculation of the partial derivatives of g(x) in relation to xk
            #
            gradxk = optimize.approx_fprime(xk, self.reliability.fel, eps, self.reliability.d)
            #
            # b. Calculation of the partial derivatives of g(x) in relation to yk
            #
            gradyk = np.transpose(Jxy).dot(gradxk)
            normgradyk = np.linalg.norm(gradyk)
            #
            # c. Calculation of the direction cosines for xk
            #
            # Direction cosines
            alpha = gradyk / normgradyk

            #
            # Step 7. Vector yk updating to yk+1 by HLRF algorithm
            #
            dk = ((np.dot(gradyk, yk) - gxk) / normgradyk ** 2) * gradyk - yk
            lambdak = 1.00
            yk1 = yk + lambdak * dk
            #
            # Parameters of iHLRF method
            #
            if iHLRF:
                gamma0 = 2.0
                a = 0.1
                # a = 0.5
                b = 0.5
                #
                gyk = gxk
                normyk = np.linalg.norm(yk)
                normyk1 = np.linalg.norm(yk1)
                c1 = normyk / normgradyk
                #
                if erro2 > delta:
                    c2 = 0.5 * normyk1 ** 2 / np.abs(gyk)
                    ck = gamma0 * np.max([c1, c2])
                else:
                    ck = gamma0 * c1
                #
                k = -1
                f1 = 1.00
                f2 = 0.00
                while f1 > f2 and k < 10:
                    k += 1
                    lambdak = b ** k
                    yk1 = yk + lambdak * dk
                    #
                    #  Transformation from yk+1 to xk+1
                    #
                    zk1 = Jzy.dot(yk1)
                    uk1 = norm.cdf(zk1)
                    for i in range(self.reliability.nxvar):
                        cdfx = uk1[i]
                        mux = mux0[i]
                        sigmax = sigmax0[i]
                        namedist = dist[i]
                        xpar1 = mux
                        xpar2 = sigmax
                        xpar3 = par3[i]
                        xpar4 = par4[i]
                        if dist[i] == 'beta':
                            xpar1 = par1[i]
                            xpar2 = par2[i]
                        if dist[i] == 'uniform':
                            xpar1 = par1[i]
                            xpar2 = par2[i]

                        xk1[i] = inverse_mapping(cdfx, xpar1, xpar2, xpar3, xpar4, namedist)

                    gyk1 = self.reliability.fel(xk1, self.reliability.d)
                    normyk1 = np.linalg.norm(yk1)
                    f1 = self.mfunc(normyk1, gyk1, ck) - self.mfunc(normyk, gyk, ck)
                    gradm = yk + ck * gradyk * np.sign(gyk)
                    normgradm = np.linalg.norm(gradm)
                    f2 = a * lambdak * np.dot(gradm, dk)
            
            #
            yk1 = yk + lambdak * dk

            #
            # Step 8. Transformation from yk+1 to xk+1
            #
            zk1 = Jzy.dot(yk1)
            uk1 = norm.cdf(zk1)
            for i in range(self.reliability.nxvar):
                cdfx = uk1[i]
                mux = mux0[i]
                sigmax = sigmax0[i]
                namedist = dist[i]
                xpar1 = mux
                xpar2 = sigmax
                xpar3 = par3[i]
                xpar4 = par4[i]
                if dist[i] == 'beta':
                    xpar1 = par1[i]
                    xpar2 = par2[i]
                if dist[i] == 'uniform':
                    xpar1 = par1[i]
                    xpar2 = par2[i]

                xk1[i] = inverse_mapping(cdfx, xpar1, xpar2, xpar3, xpar4, namedist)

            #
            # Step 9. Convergence test for yk and g(x)
            #
            prod = normgradyk * normyk
            # Evaluation of the error in the yk1 vector
            if np.abs(prod) > eps:
                erro1 = 1. - np.abs(np.dot(gradyk, yk) / (normgradyk * normyk))
            else:
                erro1 = 1000.00
            # Evaluation of the error in the limit state function g(x)
            erro2 = np.abs(gxk)
            # Printing of the updated values
            if iprint:
              DataVisualizeTransformation.one_cycle_print_results({
                'kiter': kiter,
                'gxk': gxk,
                'erro1': erro1,
                'beta': beta,
                'namevar': namevar,
                'dist': dist,
                'muxneqk': muxneqk,
                'sigmaxneqk': sigmaxneqk,
                'xk': xk,
                'yk': yk,
                'alpha': alpha
              })
        #
        pf = norm.cdf(-beta)

        if iprint: DataVisualizeTransformation.print_results({'pf': pf, 'beta': beta})
   
        return {
            "beta": beta,
            "xk": xk,
            "alpha": alpha,
            "normgradyk": normgradyk,
            } 

    def sorm(self, iHLRF=True, toler=1.e-6, iprint=True):
        """
        Second order reliability method = SORM

        """

        #
        # GRAM-SCHMIDT transformation
        #
        def gramschmidt(A, nxvar):
            rk = np.zeros(nxvar)
            rj = np.zeros(nxvar)
            rk0 = np.zeros(nxvar)
            #
            R = np.zeros((nxvar, nxvar))
            R[nxvar - 1, :] = A[nxvar - 1, :].copy()
            for k in range(nxvar - 2, -1, -1):
                rk0 = A[k, :].copy()
                rk0projection = np.zeros(nxvar)
                for j in range(nxvar - 1, k, -1):
                    rj = R[j, :].copy()
                    projection = (rj.dot(rk0)) / (rj.dot(rj))
                    rk0projection = rk0projection + projection * rj
                rk = rk0 - rk0projection
                R[k, :] = rk.copy()
            for i in range(nxvar):
                R[i, :] = R[i, :] / np.linalg.norm(R[i, :])
            #
            return R

        #
        #
        # Function to calculate the second order derivative: d2g/dxidxj
        #
        def second_order_derivative(x, i, j):
            epsilon = 1.e-4  # tolerance for the increments
            h1 = epsilon  # increments: h1 and h2, when i is not equal to j
            h2 = epsilon  # different increments can be adopted
            h = epsilon  # increment h
            a = x[i]  # reference value for x[i]
            b = x[j]  # reference value for x[j]
            #
            # Code: gmn where m and n are equal to:
            # Index 0 = no increment is applied to the variables i and j
            # Index 1 = a decrement equal to -h is applied to the variable i (or j)
            # Index 2 = an incremente equal to +h is applied to the variable i (or j)
            #
            if i == j:
                x0 = np.copy(x)
                x0[i] = a - h
                g10 = self.reliability.fel(x0, self.reliability.d)
                x0[i] = a
                g00 = self.reliability.fel(x0, self.reliability.d)
                x0[i] = a + h
                g20 = self.reliability.fel(x0, self.reliability.d)
                d2g = (g10 - 2. * g00 + g20) / h ** 2  # second order derivative: d2g/dxi2
            else:
                x0 = np.copy(x)
                x0[i] = a + h1
                x0[j] = b + h2
                g22 = self.reliability.fel(x0, self.reliability.d)
                x0[i] = a + h1
                x0[j] = b - h2
                g21 = self.reliability.fel(x0, self.reliability.d)
                x0[i] = a - h1
                x0[j] = b + h2
                g12 = self.reliability.fel(x0, self.reliability.d)
                x0[i] = a - h1
                x0[j] = b - h2
                g11 = self.reliability.fel(x0, self.reliability.d)
                d2g = (g22 - g21 - g12 + g11) / (4. * h1 * h2)  # second order derivative: d2g/dxidxj
            #
            return d2g

        #
        # First run FORM-iHLRF algorithm
        #
        n = self.reliability.nxvar
        xk = np.zeros(n)
        yk = np.zeros(n)
        gradxk = np.zeros(n)
        alpha = np.zeros(n)
        beta = 0.00
        kiter = 0
        erro1 = 0.00

        beta, xk, alpha, normgradyk, sigmaxneqk = self.form(iHLRF, toler, iprint)
        #
        # Formulation of Second Order Reliability Method - SORM
        #
        #
        # Failure probability calculation
        #
        pfform = norm.cdf(-beta)
        #
        # Calculation of the Hessian Matrix
        #
        bmatrix = np.zeros((n, n))
        dmatrix = np.zeros((n, n))
        amatrix = np.eye(n)
        hmatrix = np.zeros((n, n))

        np.set_printoptions(precision=4)

        #
        # Calculation of the Hessian matrix D: d2g/dyidyj
        #
        for i in range(n):
            for j in range(n):
                dmatrix[i, j] = second_order_derivative(xk, i, j) * sigmaxneqk[i] * sigmaxneqk[j]


        #
        # Calculation of the matrix B
        #
        bmatrix = 1. / normgradyk * dmatrix

        #
        # Calculation of the orthogonal matrix H
        #
        amatrix[n - 1, :] = alpha.copy()
        #
        hmatrix = gramschmidt(amatrix, n)


        #
        # Calculation of the curvature matrix K
        #
        kmatrix = hmatrix.dot(bmatrix.dot(hmatrix.T))

        #
        # Calculation of the failure probability using SORM Breitung equation
        #
        factor = 1.00
        for i in range(n - 1):
            factor = factor * 1. / np.sqrt(1.00 + beta * kmatrix[i, i])
        pfsorm = pfform * factor
        betasorm = -norm.ppf(pfsorm)
        #
        # Print the result
        #
        if iprint:
            print('\nSORM results:')
            print('\nHessian matrix:')
            print(dmatrix)
            print('\nNorm of the gradient of g(y) =', normgradyk)
            print('\nB matrix:')
            print(bmatrix)
            print('\nH matrix:')
            print(hmatrix)
            print('\nK = curvatures matrix:')
            print(kmatrix)
            print('\npfFORM =', pfform)
            print('\nBetaFORM =', beta)
            print('\nfactor =', factor)
            print('\npfSORM =', pfsorm)
            print('\nBetaSORM =', betasorm)

        return


