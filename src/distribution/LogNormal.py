from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np
from scipy.stats import norm

class LogNormal(AbstractDistribution):
  def __init__(self, props: dict):

    self.validate_specific_parameters(props)
    super().__init__(props)

  def validate_specific_parameters(self, props):
    ValidateDictionary.is_dictionary(props)
    ValidateDictionary.check_possible_arrays_keys(props,['varmean','varstd'], ['varmean','varcov'])
    ValidateDictionary.check_if_exists(props, 'varcov', lambda d, k: ValidateDictionary.is_greater_or_equal_than(d, k, 0))
  
  def transform(self, zk_col: np.ndarray):
    # Transform from standard normal to lognormal h(x)
    x = np.exp(self.lambdahx + zk_col * self.zetahx)

    # Compute z_f associated with target distribution f(x)
    zf = (np.log(x) - self.lambdafx) / self.zetafx

    # PDF of the original distribution f(x), modeled as normal in log space
    fx = norm.pdf(np.log(x), loc=self.lambdafx, scale=self.zetafx)

    # PDF of the sampling distribution h(x), also modeled as normal in log space
    hx = norm.pdf(np.log(x), loc=self.lambdahx, scale=self.zetahx)

    return x, fx, hx, zf