from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np
from scipy.stats import norm, lognorm
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
        """
        Transform a standard normal vector zk_col into x ~ h(x),
        and compute fx, hx, and the transformed standard normal zf.
        """
        x = self.muhx + self.sigmahx * zk_col
        fx = self.density_fx(x)
        hx = self.density_hx(x)
        zf = (np.log(x) - self.lambdafx) / self.zetafx
        return x, fx, hx, zf
    
    def sample(self, ns: int):
        """
        Sample x from the sampling distribution h(x).
        """
        return lognorm.rvs(s=self.zetahx, loc=0.0, scale=np.exp(self.lambdahx), size=ns)
    
    def density_fx(self, x: np.ndarray):
        """
        Evaluate the PDF of the target distribution f(x).
        """
        return lognorm.pdf(x, s=self.zetafx, loc=0.0, scale=np.exp(self.lambdafx))
    
    def density_hx(self, x: np.ndarray):
        """
        Evaluate the PDF of the sampling distribution h(x).
        """
        return lognorm.pdf(x, s=self.zetahx, loc=0.0, scale=np.exp(self.lambdahx))
    
    def sample_direct(self, ns: int):
        """
        Sample x from h(x), and return f(x), h(x) evaluated at x.
        This is useful for Monte Carlo sampling without transformation.
        """
        x = self.sample(ns)
        fx = self.density_fx(x)
        hx = self.density_hx(x)
        return x, fx, hx