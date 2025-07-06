import numpy as np
from scipy.stats import norm
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
import time
from scipy.optimize import curve_fit

from visualize import DataVisualizeSimulation
from .generator import RandomVariablesGenerator



class MonteCarloMethods:

  def __init__(self, parent):
      ## Get all properties about main class: Reliability
      self.reliability = parent

      ## All generator methods to random variables
      self.generator = RandomVariablesGenerator(self)


  def mc(self, nc, ns, delta_lim, nsigma=1.00, igraph=True, iprint=True):
        """
        Monte Carlo Simulation Method
        nc Cycles
        ns Simulations
        Brute force = no adaptive technique

        """
        #
        #
        ti = time.time()
        #
        # Number of variables of the problem
        #
        nc = int(nc)
        ns = int(ns)
        pfc = np.zeros(nc)
        cov_pf = np.zeros(nc)
        pf_mean = np.zeros(nc)
        sum1 = 0.00
        sum2 = 0.00
        fxmax_cycle = np.zeros(nc)
        uk_cycle = np.zeros((ns, self.reliability.nxvar))
        #
        # Correlation matrix is self.reliability.Rz
        #
        # if iprint:
        #     print('Correlation Matrix after Nataf correction:')
        #     print(self.reliability.correlation.Rz_rectify)
        #
        # Standard deviation multiplier for MC-IS
        #
        #
        nsigma = 1.00

        #
        #
        # Number of Monte Carlo simulations
        #
        #
        # Matrix xp(ns, self.reliability.nxvar) for ns Monte Carlo simulations and self.reliability.nxvar random variables
        #
        xp = np.zeros((ns, self.reliability.nxvar))
        wp = np.ones(ns)
        fx = np.ones(ns)
        zf = np.zeros((ns, self.reliability.nxvar))
        zh = np.zeros((ns, self.reliability.nxvar))

        # Matrix dmatrix(ns, self.reliability.ndvar) for ns Monte Carlo simulations and self.reliability.ndvar design variables

        dmatrix = np.array([self.reliability.d.T] * ns)

        #
        # Adaptive cycles
        #

        for icycle in range(nc):
            kcycle = icycle + 1

            #
            # Monte Carlo Simulations
            #
            #
            # Generation of uniform random numbers
            #
            index = icycle % 2
            uk_new = np.random.rand(ns, self.reliability.nxvar)
            if index == 0:
                uk_cycle = uk_new.copy()
            else:
                uk_cycle = 1.00 - uk_cycle

            #
            #
            # Step 1 - Generation of the random numbers according to their appropriate distribution
            #

            xp, wp, fx = self.generator.main(ns)
            
            #
            #
            # Step 2 - Evaluation of the limit state function g(x)
            #
            gx = list(map(self.reliability.fel, xp, dmatrix))
            gx = np.array(gx)

            #
            #
            # Step 3 - Evaluation of the indicator function I[g(x)]
            #
            igx = np.where(gx <= 0.00, wp, 0)
            nfail = sum(igx)
            pfc[icycle] = nfail/ns
            sum1 += pfc[icycle]
            sum2 += pfc[icycle] ** 2
            fxmax_cycle[icycle] = fx.max()

            #
            #  Step 6 - Evaluation of the error in the estimation of Pf
            #

            pf_mean[icycle] = sum1 / kcycle
            pf = pf_mean[icycle]
            if pf > 0.00 and kcycle > 1:
                cov_pf[icycle] = 1. / (pf * np.sqrt(kcycle * (kcycle - 1))) * np.sqrt(sum2 - 1. / kcycle * sum1 ** 2)
            else:
                cov_pf[icycle] = 0.00
            delta_pf = cov_pf[icycle]
            # Probability of failure in this cycle
            if iprint: DataVisualizeSimulation.one_cycle_print_results({'kcycle':kcycle, 'pf':pf, 'delta_pf':delta_pf})
            if delta_pf < delta_lim and kcycle > 3:
                break

        beta = -norm.ppf(pf, 0, 1)
        nsimul = kcycle * ns
        tf = time.time()
        ttotal = tf - ti
        
        # Results viewer   
        if iprint: DataVisualizeSimulation.print_results({'title':"Monte Carlo - Brute Force", 'beta': beta, 'pf':pf, 'delta_pf': delta_pf, 'nsimul': nsimul, 'gx': gx, 'ttotal': ttotal})

        if igraph: DataVisualizeSimulation.plot_results({'pf_mean':pf_mean, 'cov_pf':cov_pf, 'kcycle':kcycle})

        return {
            "beta": beta,
            "pf": pf,
            "delta_pf": delta_pf,
            "nsimul": nsimul,
            "ttotal": ttotal
            }
      
  def adaptive(self, nc, ns, delta_lim, nsigma=1.50, igraph=True, iprint=True):
      """
      Monte Carlo Simulations with Importance Sampling (MC-IS)
      Importance sampling with adaptative technique
      Melchers, R.E. Search-based importance sampling.
      Structural Safety, 9 (1990) 117-128

      """
      #
      #
      ti = time.time()
      #
      # Number of variables of the problem
      #
      nfail = 0
      niter = 0
      nc = int(nc)
      ns = int(ns)
      pfc = np.zeros(nc)
      cov_pf = np.zeros(nc)
      pf_mean = np.zeros(nc)
      sum1 = 0.00
      sum2 = 0.00
      fxmax = 0.00
      fxmax_cycle = np.zeros(nc)
      uk_cycle = np.zeros((ns, self.reliability.nxvar))

      #
      # Correlation matrix is self.reliability.Rz
      #
      # if iprint:
      #     print('Correlation Matrix after Nataf correction:')
      #     print(self.reliability.correlation.Rz_rectify)

      #
      #
      # Number of Monte Carlo simulations
      #
      #
      # Matrix xp(ns, self.reliability.nxvar) for ns Monte Carlo simulations and self.reliability.nxvar random variables
      #
      xp = np.zeros((ns, self.reliability.nxvar))
      wp = np.ones(ns)
      fx = np.ones(ns)
      zf = np.zeros((ns, self.reliability.nxvar))
      zh = np.zeros((ns, self.reliability.nxvar))


      # Matrix dmatrix(ns, self.reliability.ndvar) for ns Monte Carlo simulations and self.reliability.ndvar design variables

      dmatrix = np.array([self.reliability.d.T] * ns)

      #
      # Adaptive cycles
      #

      for icycle in range(nc):
          kcycle = icycle + 1

          #
          # Monte Carlo Simulations
          #
          #
          # Step 1 - Generation of the random numbers according to their appropriate distribution
          #

          xp, wp, fx = self.generator.main(ns)
          #
          #
          # Step 2 - Evaluation of the limit state function g(x)
          #
          gx = list(map(self.reliability.fel, xp, dmatrix))
          gx = np.array(gx)

          #
          #
          # Step 3 - Evaluation of the indicator function I[g(x)]
          #
          igx = np.where(gx <= 0.00, wp, 0)
          nfail = sum(igx)
          pfc[icycle] = nfail / ns
          sum1 += pfc[icycle]
          sum2 += pfc[icycle] ** 2
          fxmax_cycle[icycle] = fx.max()

          #
          #  Step 4 - Select adaptative mean
          #
          if nfail == 0:
              #
              # No failures in ns simulations
              #
              imin = np.argmin(gx)
              #
              i = -1
              for var in self.reliability.xvar:
                  i += 1
                  var.varhmean = xp[imin, i]

          else:
              #
              # Ocurrence of nfail failures in ns simulations
              #
              if fxmax_cycle[icycle] > 1.02 * fxmax:
                  fxmax = fxmax_cycle[icycle]
                  imax = np.argmax(fx)
                  #
                  i = -1
                  for var in self.reliability.xvar:
                      i += 1
                      var.varhmean = xp[imax, i]

          #
          #  Step 6 - Evaluation of the error in the estimation of Pf
          #

          pf_mean[icycle] = sum1 / kcycle
          pf = pf_mean[icycle]
          if pf > 0.00 and kcycle > 1:
              cov_pf[icycle] = 1. / (pf * np.sqrt(kcycle * (kcycle - 1))) * np.sqrt(sum2 - 1. / kcycle * sum1 ** 2)
          else:
              cov_pf[icycle] = 0.00
          delta_pf = cov_pf[icycle]
          # Probability of failure in this cycle
          if iprint: DataVisualizeSimulation.one_cycle_print_results({'kcycle':kcycle, 'pf':pf, 'delta_pf':delta_pf})
          if delta_pf < delta_lim and kcycle > 3:
            break

      beta = -norm.ppf(pf, 0, 1)
      nsimul = kcycle * ns
      tf = time.time()
      ttotal = tf - ti
      #
      if iprint: DataVisualizeSimulation.print_results({'title':"Monte Carlo – Adaptative Importance Sampling", 'beta': beta, 'pf':pf, 'delta_pf': delta_pf, 'nsimul': nsimul, 'gx': gx, 'ttotal': ttotal})

      if igraph: DataVisualizeSimulation.plot_results({'pf_mean':pf_mean, 'cov_pf':cov_pf, 'kcycle':kcycle})

      return {
          "beta": beta,
          "pf": pf,
          "delta_pf": delta_pf,
          "nsimul": nsimul,
          "ttotal": ttotal
      }

  def bucher(self, nc, ns, delta_lim, nsigma=1.50, igraph=True, iprint=True):
      """
      Monte Carlo Simulations with Importance Sampling (MC-IS)
      Importance sampling with adaptive technique
      BUCHER, C.G. Adaptive sampling – an iterative fast Monte Carlo procedure. Structural
      safety, v. 5, n. 2, p. 119-126, 1988.

      """
      #
      #
      ti = time.time()
      #
      # Number of variables of the problem
      #
      nc = int(nc)
      ns = int(ns)
      xm = np.zeros(self.reliability.nxvar)
      sum_xwig = np.zeros(self.reliability.nxvar)
      sum_wig = 0.00
      pfc = np.zeros(nc)
      cov_pf = np.zeros(nc)
      pf_mean = np.zeros(nc)
      sum1 = 0.00
      sum2 = 0.00
      uk_cycle = np.zeros((ns, self.reliability.nxvar))

      #
      # Correlation matrix is self.reliability.Rz
      #
      # if iprint:
      #     print('Correlation Matrix after Nataf correction:')
      #     print(self.reliability.correlation.Rz_rectify)

      #
      #
      # Number of Monte Carlo simulations
      #
      #
      # Matrix xp(ns, self.reliability.nxvar) for ns Monte Carlo simulations and self.reliability.nxvar random variables
      #
      xp = np.zeros((ns, self.reliability.nxvar))
      wp = np.ones(ns)
      fx = np.ones(ns)

      # Matrix dmatrix(ns, self.reliability.ndvar) for ns Monte Carlo simulations and self.reliability.ndvar design variables

      dmatrix = np.array([self.reliability.d.T] * ns)

      #
      # Adaptive cycles
      #

      for icycle in range(nc):
          kcycle = icycle + 1

          #
          # Monte Carlo Simulations
          #
          # Step 1 - Generation of the random numbers according to their appropriate distribution
          #

          xp, wp, fx = self.generator.main(ns)
          #
          #
          # Step 2 - Evaluation of the limit state function g(x)
          #
          gx = list(map(self.reliability.fel, xp, dmatrix))
          gx = np.array(gx)

          #
          #
          # Step 3 - Evaluation of the indicator function I[g(x)]
          #
          igx = np.where(gx <= 0.00, wp, 0)
          nfail = sum(igx)
          pfc[icycle] = nfail / ns
          sum1 += pfc[icycle]
          sum2 += pfc[icycle] ** 2
          wig = np.copy(igx)

          #
          #  Step 4 - Select adaptive mean
          #
          if nfail == 0:
              #
              # No failures in ns simulations
              #
              imin = np.argmin(gx)
              #
              i = -1
              for var in self.reliability.xvar:
                  i += 1
                  xm[i] = xp[imin, i]
                  var.varhmean = xm[i]

          else:
              #
              # Ocurrence of nfail failures in ns simulations
              #
              sum_xwig += np.dot(wig.T, xp)
              sum_wig += sum(wig)
              #
              i = -1
              for var in self.reliability.xvar:
                  i += 1
                  xm[i] = sum_xwig[i] / sum_wig
                  var.varhmean = xm[i]

          #
          #  Step 6 - Evaluation of the error in the estimation of Pf
          #

          pf_mean[icycle] = sum1 / kcycle
          pf = pf_mean[icycle]
          if pf > 0.00 and kcycle > 1:
              cov_pf[icycle] = 1. / (pf * np.sqrt(kcycle * (kcycle - 1))) * np.sqrt(sum2 - 1. / kcycle * sum1 ** 2)
          else:
              cov_pf[icycle] = 0.00
          delta_pf = cov_pf[icycle]
          nc_final = icycle
          # Probability of failure in this cycle
          if iprint: DataVisualizeSimulation.one_cycle_print_results({'kcycle':kcycle, 'pf':pf, 'delta_pf':delta_pf})
          if delta_pf < delta_lim and kcycle > 3:
            break

      beta = -norm.ppf(pf, 0, 1)
      nsimul = kcycle * ns
      tf = time.time()
      ttotal = tf - ti
      #
      if iprint: DataVisualizeSimulation.print_results({'title':"Monte Carlo – Bucher Importance Sampling", 'beta': beta, 'pf':pf, 'delta_pf': delta_pf, 'nsimul': nsimul, 'gx': gx, 'ttotal': ttotal})

      if igraph: DataVisualizeSimulation.plot_results({'pf_mean':pf_mean, 'cov_pf':cov_pf, 'kcycle':kcycle})


      return {
          "beta": beta,
          "pf": pf,
          "delta_pf": delta_pf,
          "nsimul": nsimul,
          "ttotal": ttotal
      }

  def sampling_project_point(self, nc, ns, delta_lim, igraph=True, iprint=True):   

        ti = time.time()
        nc = int(nc)
        ns = int(ns)
        pfc = np.zeros(nc)
        sum1 = 0.00
        sum2 = 0.00
        pf_mean = np.zeros(nc)
        cov_pf = np.zeros(nc)

        ## Apriori Results
        formResults = self.reliability.form(iHLRF=True, toler=1.e-6)

        ## Project Point
        xk = formResults['xk']
        self.reliability.x0 = xk

        ## Using varhmean calculate based on x0
        for var, mean_value in zip(self.reliability.xvar, self.reliability.x0):
          var['varhmean'] = mean_value

        # Matrix dmatrix(ns, self.reliability.ndvar) for ns Monte Carlo simulations and self.reliability.ndvar design variables
        dmatrix = np.array([self.reliability.d.T] * ns)

        for icycle in range(nc):
          kcycle = icycle + 1

          # Monte Carlo Simulations

          # Step 1 - Generation of the random numbers according to their appropriate distribution
          xp, wp, fx = self.generator.main(ns)

          # Step 2 - Evaluation of the limit state function g(x)
          gx = list(map(self.reliability.fel, xp, dmatrix))
          gx = np.array(gx)

          # Step 3 - Evaluation of the indicator function I[g(x)]
          igx = np.where(gx <= 0.00, wp, 0)
          nfail = sum(igx)
          pfc[icycle] = nfail / ns
          sum1 += pfc[icycle]
          sum2 += pfc[icycle] ** 2
        
          # Step 4 - Evaluation of the error in the estimation of Pf
          pf_mean[icycle] = sum1 / kcycle
          
          pf = pf_mean[icycle]
          if pf > 0.00 and kcycle > 1:
              cov_pf[icycle] = 1. / (pf * np.sqrt(kcycle * (kcycle - 1))) * np.sqrt(sum2 - 1. / kcycle * sum1 ** 2)
          else:
              cov_pf[icycle] = 0.00

          delta_pf = cov_pf[icycle]
          # Plot probability of failure in this cycle
          if iprint: DataVisualizeSimulation.one_cycle_print_results({'kcycle':kcycle, 'pf':pf, 'delta_pf':delta_pf})
          if delta_pf < delta_lim and kcycle > 3:
              break

        beta = -norm.ppf(pf, 0, 1)
        nsimul = kcycle * ns
        tf = time.time()
        ttotal = tf - ti

        # Results viewer
        
        if iprint: DataVisualizeSimulation.print_results({'title':"Monte Carlo – Importance Sampling Based on the Design Point", 'beta': beta, 'pf':pf, 'delta_pf': delta_pf, 'nsimul': nsimul, 'gx': gx, 'ttotal': ttotal})
        if igraph: DataVisualizeSimulation.plot_results({'pf_mean':pf_mean, 'cov_pf':cov_pf, 'kcycle':kcycle})

        return {
              "beta": beta,
              "pf": pf,
              "delta_pf": delta_pf,
              "nsimul": nsimul,
              "ttotal": ttotal
          }

  def sampling_enhanced(self, nc, ns, delta_lim, igraph=True, iprint=True):

    def calculate_pf(arr_pf, arr_lambda):
          def pf_model(lam, a, b, c, q):
              base = np.maximum(lam - b, 0)
              return q * np.exp(-a * base**c)

          # Initial parameters and limits
          initial_guess = [5.0, 0.1, 1.0, max(arr_pf)]
          bounds = ([0.001, -1.0, 0.1, 0.001], [100.0, 1.5, 5.0, 1.0])

          try:
              params, _ = curve_fit(pf_model, arr_lambda, arr_pf, p0=initial_guess, bounds=bounds, maxfev=20000)
              pf_at_cycle = pf_model(1.0, *params)
              return pf_at_cycle
          
          except Exception as e:
              print("Error in adjustment:", e)
              return np.nan 
                    
    def linspace_between_0_and_1(n):
      """
      Returns a NumPy array with n equally spaced values between 0 and 1,
      excluding 0 and 1.

      Parameters:

      n: int, the number of desired subdivisions in the interval (0, 1)

      Returns:

      numpy.ndarray of floats between 0 and 1, excluding the endpoints.
      """
      return np.linspace(0, 1, n + 2)[1:-1]
    
    ti = time.time()
    nc = int(nc)
    ns = int(ns)

    lambdas = linspace_between_0_and_1(100)
    nlambdas = len(lambdas)

    pfc = np.zeros((nc, nlambdas))
    pf_mean = np.zeros(nc)
    cov_pf = np.zeros(nc)
    sum1 = 0.00
    sum2 = 0.00

    varhmean_array = [var.varhmean for var in self.reliability.xvar]
    dvar_array = [var['varvalue'] for var in self.reliability.dvar]

    gx_based_varhmean = self.reliability.fel(varhmean_array, dvar_array)

    deduction = (1 - lambdas) * gx_based_varhmean
    deductions = np.tile(deduction[:, None], (1, ns))

    # Matrix dmatrix(ns, self.reliability.ndvar) for ns Monte Carlo simulations and self.reliability.ndvar design variables
    dmatrix = np.array([self.reliability.d.T] * ns)

    for icycle in range(nc):
      kcycle = icycle + 1

      # Monte Carlo Simulations
      # Generation of uniform random numbers - Antithetic Sampling
      #
      index = icycle % 2
      uk_new = np.random.rand(ns, self.reliability.nxvar)
      if index == 0:
          uk_cycle = uk_new.copy()
      else:
          uk_cycle = 1.00 - uk_cycle

      # Step 1 - Generation of the random numbers according to their appropriate distribution
      xp, wp, fx = self.generator.main(ns)

      # Step 2 - Evaluation of the limit state function g(x)
      gx = list(map(self.reliability.fel, xp, dmatrix))
      gx = np.array(gx)

      # Step 3 - Evaluation of the new limit state function m(x)
      gx_lambdas = np.tile(gx, (nlambdas, 1))
      mx_lambdas = gx_lambdas - deductions

      #Step 4 - Evaluation of the indicator function I[g(x)]
      imx_lambdas = np.where(mx_lambdas <= 0.00, wp, 0)
      nfail = np.sum(imx_lambdas, axis=1)
      pfc[icycle] = nfail / ns       

      #Add the point lambda = 0 and Pf = 0.5
      arr_lambda = np.insert(lambdas, 0, 0.0)
      arr_pf = np.insert(pfc[icycle], 0, 0.5)
      pf_cycle = calculate_pf(arr_pf, arr_lambda)

      sum1 += pf_cycle
      sum2 += pf_cycle ** 2

      pf_mean[icycle] = sum1 / kcycle
      pf = pf_mean[icycle]

      if pf > 0.00 and kcycle > 1:
          cov_pf[icycle] = 1. / (pf * np.sqrt(kcycle * (kcycle - 1))) * np.sqrt(sum2 - 1. / kcycle * sum1 ** 2)
      else:
          cov_pf[icycle] = 0.00

      delta_pf = cov_pf[icycle]
      
      # Plot probability of failure in this cycle
      if iprint: DataVisualizeSimulation.one_cycle_print_results({'kcycle':kcycle, 'pf':pf, 'delta_pf':delta_pf})
      if delta_pf < delta_lim and kcycle > 3:
        break

    beta = -norm.ppf(pf, 0, 1)
    nsimul = kcycle * ns
    tf = time.time()
    ttotal = tf - ti

    # Results viewer   
    if iprint: DataVisualizeSimulation.print_results({'title':"Monte Carlo – Enhanced Importance Sampling", 'beta': beta, 'pf':pf, 'delta_pf': delta_pf, 'nsimul': nsimul, 'gx': gx, 'ttotal': ttotal})
    if igraph: DataVisualizeSimulation.plot_results({'pf_mean':pf_mean, 'cov_pf':cov_pf, 'kcycle':kcycle})

    return {
          "beta": beta,
          "pf": pf,
          "delta_pf": delta_pf,
          "nsimul": nsimul,
          "ttotal": ttotal
      }
              