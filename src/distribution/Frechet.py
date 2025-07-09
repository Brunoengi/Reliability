from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
from scipy.special import gamma
from scipy.optimize import newton
from scipy.stats import invweibull, norm
import numpy as np

class Frechet(AbstractDistribution):
    def __init__(self, props: dict):
        self.validate_specific_parameters(props)
        super().__init__(props)

        # Calculates parameters of the target distribution f(x)
        deltafx = self.sigmafx / self.mufx
        self.kapaf = newton(self.fkapa, 2.5, args=(deltafx, -1))
        self.vfn = self.mufx / gamma(1 - 1 / self.kapaf)

        # Initialize parameters of h(x)
        self.update_sampling(nsigma=1.0)

    def validate_specific_parameters(self, props):
        ValidateDictionary.is_dictionary(props)
        ValidateDictionary.has_keys(props, 'varmean', 'varcov')
        ValidateDictionary.check_if_exists(props, 'varcov', lambda d, k: ValidateDictionary.is_greater_or_equal_than(d, k, 0))

    def fkapa(self, kapa, delta, gsignal):
        """Function to find the root in determining the shape parameter kapa."""
        return 1 + delta**2 - gamma(1 + 2 * gsignal / kapa) / (gamma(1 + gsignal / kapa) ** 2)

    def update_sampling(self, nsigma: float = 1.0):
        """Updates the parameters of the sampling distribution h(x) based on nsigma."""
        self.sigmahx = self.sigmafx * nsigma
        self.muhx = self.mufx

        deltahx = self.sigmahx / self.muhx
        self.kapah = newton(self.fkapa, 2.5, args=(deltahx, -1))
        self.vhn = self.muhx / gamma(1 - 1 / self.kapah)

    def transform(self, zk_col: np.ndarray):
      """
      Transforms a standard normal variable zk_col into the target distribution f(x),
      and returns the transformed x, fx, hx, and zf.
      """
      uk = norm.cdf(zk_col)
      x = self.vhn / (np.log(1 / uk)) ** (1 / self.kapah)

      cdfx = invweibull.cdf(x / self.vfn, self.kapaf)
      zf = norm.ppf(cdfx)

      fx = self.density_fx(x)
      hx = self.density_hx(x)

      return x, fx, hx, zf
    
    def sample(self, ns: int):
        """
        Sample values ​​of x from the sampling distribution h(x).
        """
        u = np.random.rand(ns)
        x = self.vhn / (np.log(1 / u)) ** (1 / self.kapah)
        return x

    def density_fx(self, x: np.ndarray):
        """
        Evaluates the density of the target distribution f(x) at points x.
        """
        y = x / self.vfn
        return invweibull.pdf(y, self.kapaf) / self.vfn

    def density_hx(self, x: np.ndarray):
        """
        Evaluates the density of the sampling distribution h(x) at points x.
        """
        y = x / self.vhn
        return invweibull.pdf(y, self.kapah) / self.vhn

    def sample_direct(self, ns: int):
        """     
        Sample x ~ h(x) and compute fx, hx at the sampled points.
        Returns: x, fx, hx
        """
        x = self.sample(ns)
        fx = self.density_fx(x)
        hx = self.density_hx(x)
        return x, fx, hx