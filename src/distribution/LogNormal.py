from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np
from scipy.stats import norm
from math import sqrt, log

class LogNormal(AbstractDistribution):
    def __init__(self, props: dict):
        self.validate_specific_parameters(props)
        super().__init__(props)

        # Parameters for f(x) (target distribution)
        self.zetafx = sqrt(log(1 + (self.sigmafx / self.mufx) ** 2))
        self.lambdafx = log(self.mufx) - 0.5 * self.zetafx ** 2

        # Initialize parameters of h(x)
        self.update_sampling(nsigma=1.0)

    def validate_specific_parameters(self, props):
        ValidateDictionary.is_dictionary(props)
        ValidateDictionary.check_possible_arrays_keys(props,['varmean','varstd'], ['varmean','varcov'])
        ValidateDictionary.check_if_exists(props, 'varcov', lambda d, k: ValidateDictionary.is_greater_or_equal_than(d, k, 0))
    
    def update_sampling(self, nsigma: float = 1.0):
        "Updates the parameters of h(x) based on nsigma."
        self.muhx = self.mufx
        self.sigmahx = self.sigmafx * nsigma
        self.zetahx = sqrt(log(1 + (self.sigmahx / self.muhx) ** 2))
        self.lambdahx = log(self.muhx) - 0.5 * self.zetahx ** 2

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