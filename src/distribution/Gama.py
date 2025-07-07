from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
from scipy.stats import gamma as gamma_dist, norm
import numpy as np


class Gama(AbstractDistribution):
    def __init__(self, props: dict):
        self.validate_specific_parameters(props)
        super().__init__(props)

        # Parameters of the target distribution f(x)
        self.deltafx = self.sigmafx / self.mufx
        self.k = 1 / self.deltafx**2
        self.v = self.k / self.mufx
        self.a = self.k
        self.loc = 0
        self.scale = 1 / self.v

        # Initialize sampling distribution h(x) parameters
        self.update_sampling(nsigma=1.0)

    def validate_specific_parameters(self, props):
        ValidateDictionary.is_dictionary(props)
        ValidateDictionary.check_possible_arrays_keys(props, ['varmean', 'varcov'], ['varmean', 'varstd'])
        ValidateDictionary.check_if_exists(props, 'varcov', lambda d, k: ValidateDictionary.is_greater_or_equal_than(d, k, 0))

    def update_sampling(self, nsigma: float = 1.0):
        """Update sampling distribution h(x) parameters based on nsigma."""
        self.muhx = self.mufx
        self.sigmahx = self.sigmafx * nsigma

        self.deltahx = self.sigmahx / self.muhx
        self.kh = 1 / self.deltahx**2
        self.vh = self.kh / self.muhx
        self.ah = self.kh
        self.loch = 0
        self.scaleh = 1 / self.vh

    def sample(self, n: int):
        """Draw n samples from the importance distribution h(x)."""
        return gamma_dist.rvs(self.ah, loc=self.loch, scale=self.scaleh, size=n)

    def density_fx(self, x: np.ndarray):
        """Evaluate the target PDF f(x) at x."""
        return gamma_dist.pdf(x, self.a, loc=self.loc, scale=self.scale)

    def density_hx(self, x: np.ndarray):
        """Evaluate the importance PDF h(x) at x."""
        return gamma_dist.pdf(x, self.ah, loc=self.loch, scale=self.scaleh)

    def sample_direct(self, n: int):
        """
        Draw n samples directly from the target distribution f(x)
        and evaluate f(x) and h(x) at those points.
        """
        x = gamma_dist.rvs(self.a, loc=self.loc, scale=self.scale, size=n)
        fx = self.density_fx(x)
        hx = self.density_hx(x)
        return x, fx, hx

    def transform(self, zk_col: np.ndarray):
        """
        Transform standard normal samples to physical space using h(x),
        then compute corresponding densities and map to standard normal space using f(x).
        """
        uk = norm.cdf(zk_col)

        # Transform from standard normal space to physical space using h(x)
        x = gamma_dist.ppf(uk, self.ah, loc=self.loch, scale=self.scaleh)

        # Evaluate densities using existing methods
        fx = self.density_fx(x)
        hx = self.density_hx(x)

        # Map back to standard normal space using f(x)
        cdfx = gamma_dist.cdf(x, self.a, loc=self.loc, scale=self.scale)
        zf = norm.ppf(cdfx)

        return x, fx, hx, zf