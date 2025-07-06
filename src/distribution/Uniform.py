from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm, uniform

class Uniform(AbstractDistribution):
  def __init__(self, props):

    if(set(['parameter1','parameter2']).issubset(props)):
      self.a = props['parameter1']
      self.b = props['parameter2']

      self.varmean = float((self.a + self.b) / 2)
      self.varstd = float((self.b - self.a) / np.sqrt(12))
      self.varhmean = float(self.varmean)

    elif set(['varmean', 'varstd']).issubset(props):
      # Case: varmean and varstd are provided — compute a and b
      self.varmean = float(props['varmean'])
      self.varstd = float(props['varstd'])
      self.varhmean = self.varmean

      # Solve for a and b given mean and std of a uniform distribution
      self.a, self.b = fsolve(self.uniform_limits, (self.varmean - 1, self.varmean + 1), args=(self.varmean, self.varstd))
    
    else:
      raise ValueError("Uniform distribution requires either ['parameter1', 'parameter2'] or ['varmean', 'varstd'].")

    props.update({
            'varmean': self.varmean,
            'varstd': self.varstd,
            'varhmean': self.varhmean,
            'parameter1': self.a,
            'parameter2': self.b
        })

    super().__init__(props)

  def validate_specific_parameters(self, props):
    ValidateDictionary.check_possible_arrays_keys(props, ['varmean','varstd'], ['parameter1', 'parameter2'])
  
  @staticmethod
  def uniform_limits(vars, mux, sigmax):
    a_, b_ = vars
    eq1 = (a_ + b_) / 2 - mux
    eq2 = (b_ - a_) / np.sqrt(12) - sigmax
    return [eq1, eq2]
  
  def transform(self, zk_col: np.ndarray):
    # Compute the limits a_h and b_h of the sampling distribution h(x) 
    # such that it has the same mean and standard deviation as specified
    ah, bh = fsolve(self.uniform_limits, (self.a, self.b), args=(self.varhmean, self.varstd))

    # Transform from standard normal to uniform using the inverse CDF method
    uk = norm.cdf(zk_col)
    x = ah + (bh - ah) * uk

    # Inverse transformation to z_f (standard normal variable associated with f(x))
    zf = norm.ppf(uk)

    # PDF of the original distribution f(x)
    fx = uniform.pdf(x, loc=self.a, scale=self.b - self.a)

    # PDF of the sampling distribution h(x)
    hx = uniform.pdf(x, loc=ah, scale=bh - ah)

    return x, fx, hx, zf