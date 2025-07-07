import scipy.optimize
import numpy as np
import scipy.linalg
from math import log
from scipy.stats import norm, uniform, lognorm, gumbel_r, invweibull, weibull_min, beta as beta_dist, gamma as gamma_dist, multivariate_normal
from scipy.optimize import fsolve, newton
from scipy.linalg import cholesky
from math import sqrt, pi, log
from scipy.special import gamma


class RandomVariablesGenerator:
  def __init__(self, parent):
      ## Get all properties about Monte Carlo Methods
      self.parent = parent

      ## Get all properties about Reliability
      self.reliability = parent.reliability

  def main(self, ns, nsigma=1.00, iprint=False):
        """
        Method to generate random variables, check correlation matrix to use var_gen (correlated variables) or var_rvs (uncorrelated variables )

        """
        # Number of variables of the problem
        ns = int(ns)
        
        # Get index of correlated and uncorrelated variables
        index_correlated, index_uncorrelated = self.reliability.correlation.correlation_summary()
        total_index = len(index_correlated) + len(index_uncorrelated) 

        # Empty matrix 
        x = np.empty((ns, total_index))

        #
        # Standard deviation multiplier for MC-IS
        
        #
        # Step 1 - Generation of the random numbers according to their appropriate distribution
        #

        wpc = np.ones(ns)
        fxc = np.ones(ns)
        wpu = np.ones(ns)
        fxu = np.ones(ns)
        
        if index_correlated:
          xpc, wpc, fxc = self.var_gen(ns, index_correlated, nsigma)
        
        if index_uncorrelated:
          xpu, wpu, fxu = self.var_rvs(ns, index_uncorrelated, nsigma)
        


        for i, idx in enumerate(index_correlated):
          x[:, idx] = xpc[:, i]

        # Preencher as colunas dos não correlacionados
        if index_correlated:
          for i, idx in enumerate(index_correlated):
            x[:, idx] = xpc[:, i]

        if index_uncorrelated:
          for i, idx in enumerate(index_uncorrelated):
            x[:, idx] = xpu[:, i]

        wp = wpc * wpu
        fx = fxc * fxu

        return x, wp, fx

  def var_gen(self, ns, indexes_correlated_xvar, nsigma=1.00, iprint=False):
      """
      Random variables generator for Monte Carlo Simulation methods, only to correlated variables
      """
      xvar_correlated = [self.reliability.xvar[i] for i in indexes_correlated_xvar]
      nxvar_correlated = len(xvar_correlated)

      # Generate initial matrices and variables
      uk_cycle = np.random.rand(ns, nxvar_correlated)
      x = np.zeros((ns, nxvar_correlated))
      weight = np.ones(ns)
      fxixj = np.ones(ns)
      zf = np.zeros((ns, nxvar_correlated))

      # Correlation submatrix for correlated variables
      matrix = self.reliability.correlation.Rz_rectify[np.ix_(indexes_correlated_xvar, indexes_correlated_xvar)]

      # Cholesky to generate correlated Gaussian samples
      L = scipy.linalg.cholesky(matrix, lower=True)
      yk = norm.ppf(uk_cycle)       # Independent standard normals
      zk = yk @ L.T                 # Correlated normals

      for i, var in enumerate(xvar_correlated):
          zk_col = zk[:, i]
          x[:, i], fx, hx, zf[:, i] = var.transform(zk_col)

          # Update weights and fxixj for variable i
          w, fx_over_norm = var.update_weights(fx, hx, zf[:, i], zk_col)
          weight *= w
          fxixj *= fx_over_norm

      # Final correction with multivariate PDFs
      norm_multivar = multivariate_normal(cov=matrix)
      phif = norm_multivar.pdf(zf)
      phih = norm_multivar.pdf(zk)

      weight *= phif / phih
      fxixj *= phif

      return x, weight, fxixj 
  
  def var_rvs(self, ns, indexes_uncorrelated_xvar, nsigma=1.00, iprint=False):
    """

        Random variables generator for the Monte Carlo Simulation methods, only to uncorrelated variables

    """

    #Get only correlated variables
    xvar_uncorrelated = [self.reliability.xvar[i] for i in indexes_uncorrelated_xvar]
    nxvar_uncorrelated = len(xvar_uncorrelated)

    def fkapa(kapa, deltax, gsignal):
        fk = 1.00 + deltax ** 2 - gamma(1.00 + gsignal * 2.00 / kapa) / gamma(1.00 + gsignal * 1.00 / kapa) ** 2
        return fk
    
    def beta_limits(vars, mux, sigmax, q, r):
        a, b = vars
        eq1 = a + q / (q + r) * (b - a) - mux
        eq2 = ((q * r) / ((q + r) ** 2 * (q + r + 1))) ** (0.50) * (b - a) - sigmax
        return [eq1, eq2]
    
    def uniform_limits(vars, mux, sigmax):
        a, b = vars
        eq1 = (a + b) / 2 - mux
        eq2 = (b - a) / np.sqrt(12.) - sigmax
        return [eq1, eq2]

    

    x = np.zeros((ns, nxvar_uncorrelated))
    weight = np.ones(ns)
    fx = np.zeros(ns)
    hx = np.zeros(ns)
    fxixj = np.ones(ns)
    
    #
    # Step 1 - Determination of equivalent correlation coefficients and
    #          Jacobian matrix Jzy
    #
    #
    # Cholesky decomposition of the correlation matrix
    #

    #L = scipy.linalg.cholesky(self.reliability.correlation.Rz_rectify, lower=True)
    #Jzy = np.copy(L)

    #
    # Generation of Gaussian random numbers
    #

    
    zf = np.zeros((ns, nxvar_uncorrelated))
    
    #
    i = -1
    for var in xvar_uncorrelated:
        i += 1
        if var.varstd == 0.00:
            var.varstd = float(var.varcov) * float(var.varmean)
        if iprint:
            print(self.reliability.xvar[i])
        #
        #
        # Normal distribution
        #
        namedist = var.namedist
        mufx = var.mufx
        sigmafx = var.sigmafx
        muhx = var.muhx
        sigmahx = nsigma * sigmafx

        if namedist.lower() == 'gauss':
            x[:, i] = norm.rvs(loc=muhx, scale=sigmahx, size=ns)
            fx = norm.pdf(x[:, i], mufx, sigmafx)
            hx = norm.pdf(x[:, i], muhx, sigmahx)
            weight = weight * (fx / hx)
            fxixj = fxixj * fx 
        #
        # Uniform or constant distribution
        #
        
        elif namedist.lower() == 'uniform':
            a = float(var.parameter1)
            b = float(var.parameter2)
            ah, bh =  fsolve(uniform_limits, (1, 1), args= (muhx, sigmahx))  
                            
            
            x[:, i] = uniform.rvs(loc=ah, scale= (bh-ah), size = ns)
            fx = uniform.pdf(x[:, i], a, b-a)
            hx = uniform.pdf(x[:, i], ah, bh-ah)
            weight = weight * (fx / hx)
            fxixj = fxixj * fx 
        #
        # Lognormal distribution
        #
        elif namedist.lower() == 'lognorm':
            zetafx = np.sqrt(np.log(1.00 + (sigmafx / mufx) ** 2))
            lambdafx = np.log(mufx) - 0.5 * zetafx ** 2
            zetahx = np.sqrt(np.log(1.00 + (sigmahx / muhx) ** 2))
            lambdahx = np.log(muhx) - 0.5 * zetahx ** 2
            x[:, i] = lognorm.rvs(s=zetahx, loc=0.00, scale=np.exp(lambdahx), size=ns)
            fx = lognorm.pdf(x[:, i], s=zetafx, loc=0.00, scale=np.exp(lambdafx))
            hx = lognorm.pdf(x[:, i], s=zetahx, loc=0.00, scale=np.exp(lambdahx))
            weight = weight * (fx / hx)
            fxixj = fxixj * fx 

        #
        # Gumbel distribution
        #
        elif namedist.lower() == 'gumbel':
            sigmahx = nsigma * sigmafx
            alphafn = np.pi / np.sqrt(6.00) / sigmafx
            ufn = mufx - np.euler_gamma / alphafn
            betafn = 1.00 / alphafn
            alphahn = np.pi / np.sqrt(6.00) / sigmahx
            uhn = muhx - np.euler_gamma / alphahn
            betahn = 1.00 / alphahn
            x[:, i] = gumbel_r.rvs( loc=uhn, scale=betahn, size=ns)
            fx = gumbel_r.pdf(x[:, i], ufn, betafn)
            hx = gumbel_r.pdf(x[:, i], uhn, betahn)
            weight = weight * (fx / hx)
            fxixj = fxixj * fx 

        #
        # Frechet distribution
        #
        elif namedist.lower() == 'frechet':
            deltafx = sigmafx / mufx
            kapa0 = 2.50
            gsinal = -1.00
            kapaf = scipy.optimize.newton(fkapa, kapa0, args=(deltafx, gsinal))
            vfn = mufx / gamma(1.00 - 1.00 / kapaf)
            deltahx = sigmahx / muhx
            kapa0 = 2.50
            gsinal = -1.00
            kapah = scipy.optimize.newton(fkapa, kapa0, args=(deltahx, gsinal))
            vhn = muhx / gamma(1.00 - 1.00 / kapah)
            x[:, i] = invweibull.rvs(c=kapah, loc=0.00, scale=vhn, size=ns)
            fx = invweibull.pdf(x[:, i], c=kapaf, loc=0.00, scale=vfn)
            hx = invweibull.pdf(x[:, i], c=kapah, loc=0.00, scale=vhn)
            weight = weight * (fx / hx)
            fxixj = fxixj * fx 

        #
        #
        # Weibull distribution - minimum
        #
        elif namedist.lower() == 'weibull':
            epsilon = float(var.epsilon)
            deltafx = sigmafx / (mufx - epsilon)
            kapa0 = 2.50
            gsinal = 1.00
            kapaf = scipy.optimize.newton(fkapa, kapa0, args=(deltafx, gsinal))
            w1f = (mufx - epsilon) / gamma(1.00 + 1.00 / kapaf) + epsilon
            deltahx = sigmahx / (muhx - epsilon)
            kapa0 = 2.50
            gsinal = 1.00
            kapah = scipy.optimize.newton(fkapa, kapa0, args=(deltahx, gsinal))
            w1h = (muhx - epsilon) / gamma(1.00 + 1.00 / kapah) + epsilon
            x[:, i] = weibull_min.rvs(c=kapah, loc=epsilon, scale=w1h-epsilon, size=ns)
            fx = weibull_min.pdf(x[:, i], c=kapaf, loc=epsilon, scale=w1f-epsilon)
            hx = weibull_min.pdf(x[:, i], c=kapah, loc=epsilon, scale=w1h-epsilon)
            weight = weight * (fx / hx)
            fxixj = fxixj * fx 

        #
        #
        # Beta distribution
        #
        elif namedist.lower() == 'beta':
            
           
            loc = var.a
            scale = (var.b - var.a)
            
            
            ah, bh =  fsolve(beta_limits, (1, 1), args= ( muhx, sigmahx, var.q, var.r))  
            loch = ah
            scaleh = (bh - ah)        
            x[:, i] = beta_dist.rvs(var.q, var.r, loch, scaleh, size=ns)
            fx = beta_dist.pdf(x[:, i], var.q, var.r, loc, scale)
            hx = beta_dist.pdf(x[:, i], var.q, var.r, loch, scaleh)
            weight = weight * (fx / hx)
            fxixj = fxixj * fx 

        #
        #
        # Gamma distribution
        #
        elif namedist.lower() == 'gamma':
            
            deltafx = sigmafx / mufx
            k = 1. / deltafx ** 2
            v = k / mufx
            a = k
            loc = 0.00
            scale = 1. / v
          
            deltahx = sigmahx / muhx
            kh = 1. / deltahx ** 2
            vh = kh / muhx
            ah = kh
            loch = 0.00
            scaleh = 1. / vh
            x[:, i] = gamma_dist.rvs(ah, loch, scaleh, size=ns)
            fx = gamma_dist.pdf(x[:, i], a, loc, scale)
            hx = gamma_dist.pdf(x[:, i], ah, loch, scaleh)
            weight = weight * (fx / hx)
            fxixj = fxixj * fx 
            

    return x, weight, fxixj
  

