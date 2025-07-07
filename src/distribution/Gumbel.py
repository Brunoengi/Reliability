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
        self.alphaf = pi / sqrt(6.0) / self.sigmafx
        self.ufx = self.mufx - self.euler_gamma / self.alphaf
        self.betafx = 1.0 / self.alphaf

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

        self.alphah = pi / sqrt(6.0) / self.sigmahx
        self.uhx = self.muhx - self.euler_gamma / self.alphah
        self.betahx = 1.0 / self.alphah

    def transform(self, zk_col: np.ndarray):
        """
        Transform a standard normal vector zk_col into x ~ h(x),
        and compute fx, hx, and the transformed standard normal zf.
        """
        if not hasattr(self, 'uhx') or not hasattr(self, 'betahx'):
            raise RuntimeError("Sampling distribution parameters not defined. Call update_sampling(nsigma) first.")

        uk = norm.cdf(zk_col)
        x = self.uhx - self.betahx * np.log(-np.log(uk))

        fx = self.density_fx(x)
        hx = self.density_hx(x)
        zf = norm.ppf(gumbel_r.cdf(x, loc=self.ufx, scale=self.betafx))
        return x, fx, hx, zf

    def sample(self, ns: int):
        """
        Sample x from the sampling distribution h(x).
        """
        return gumbel_r.rvs(loc=self.uhx, scale=self.betahx, size=ns)

    def density_fx(self, x: np.ndarray):
        """
        Evaluate the PDF of the target distribution f(x).
        """
        return gumbel_r.pdf(x, loc=self.ufx, scale=self.betafx)

    def density_hx(self, x: np.ndarray):
        """
        Evaluate the PDF of the sampling distribution h(x).
        """
        return gumbel_r.pdf(x, loc=self.uhx, scale=self.betahx)

    def sample_direct(self, ns: int):
        """
        Sample x from h(x), and return f(x), h(x) evaluated at x.
        Useful for direct sampling in Monte Carlo methods.
        """
        x = self.sample(ns)
        fx = self.density_fx(x)
        hx = self.density_hx(x)
        return x, fx, hx