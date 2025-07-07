from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
from scipy.stats import gumbel_r, norm
from numpy import sqrt, pi, log
import numpy as np

class Gumbel(AbstractDistribution):
  def __init__(self, props: dict):
    self.validate_specific_parameters(props)
    super().__init__(props)

    # Euler–Mascheroni constant
    self.euler_gamma = 0.5772156649015329

    # Parameters of the target distribution f(x)
    self.alphaf = pi / sqrt(6) / self.sigmafx
    self.ufx = self.mufx - self.euler_gamma / self.alphaf
    self.betafx = 1 / self.alphaf

    # Initialize h(x) with the same parameters as f(x)
    self.update_sampling(nsigma=1.0)

  def validate_specific_parameters(self, props):
      ValidateDictionary.is_dictionary(props)
      ValidateDictionary.check_possible_arrays_keys(props, ['varmean', 'varstd'], ['varmean', 'varcov'])
      ValidateDictionary.check_if_exists(props, 'varcov', lambda d, k: ValidateDictionary.is_greater_or_equal_than(d, k, 0))

  def update_sampling(self, nsigma: float = 1.0):
      """Updates the parameters of the sampling distribution h(x) based on nsigma."""
      self.sigmahx = self.sigmafx * nsigma
      self.muhx = self.mufx

      self.alphah = pi / sqrt(6) / self.sigmahx
      self.uhx = self.muhx - self.euler_gamma / self.alphah
      self.betahx = 1 / self.alphah

  def transform(self, zk_col: np.ndarray):
    if not hasattr(self, 'uhx') or not hasattr(self, 'betahx'):
      raise RuntimeError("Parâmetros de h(x) não definidos. Chame update_sampling(nsigma).")

    # Transform standard variable zk to Gumbel distribution h(x)
    uk = norm.cdf(zk_col)  # u ~ Uniform(0,1)
    x = self.uhx - self.betahx * log(log(1 / uk))

    # Calculates zf based on the target distribution f(x)
    cdf_fx = gumbel_r.cdf(x, loc=self.ufx, scale=self.betafx)
    zf = norm.ppf(cdf_fx)

    # Calculates densities f(x) and h(x)
    fx = gumbel_r.pdf(x, loc=self.ufx, scale=self.betafx)
    hx = gumbel_r.pdf(x, loc=self.uhx, scale=self.betahx)

    return x, fx, hx, zf