from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
from scipy.special import gamma
from scipy.optimize import newton
from scipy.stats import weibull_min, norm
import numpy as np

class Weibull(AbstractDistribution):
    def __init__(self, props: dict):
        self.validate_specific_parameters(props)
        super().__init__(props)

        self.epsilon = float(props['parameter3'])

        # Parameters of the target distribution f(x)
        deltafx = self.sigmafx / (self.mufx - self.epsilon)
        self.kapaf = newton(self.fkapa, 2.5, args=(deltafx, 1))
        self.w1f = (self.mufx - self.epsilon) / gamma(1 + 1 / self.kapaf) + self.epsilon

        # Initialize sampling distribution h(x) parameters
        self.update_sampling(nsigma=1.0)

    def validate_specific_parameters(self, props):
        ValidateDictionary.is_dictionary(props)
        ValidateDictionary.has_keys(props, 'varmean', 'varstd', 'parameter3')
        ValidateDictionary.check_if_exists(props, 'varcov', lambda d, k: ValidateDictionary.is_greater_or_equal_than(d, k, 0))

    def fkapa(self, kapa, delta, gsignal):
        """Function to find root for the shape parameter kapa."""
        return 1 + delta**2 - gamma(1 + gsignal * 2 / kapa) / (gamma(1 + gsignal / kapa) ** 2)

    def update_sampling(self, nsigma: float = 1.0):
        """Update sampling distribution h(x) parameters based on nsigma."""
        self.sigmahx = self.sigmafx * nsigma
        self.muhx = self.mufx

        deltahx = self.sigmahx / (self.muhx - self.epsilon)
        self.kapah = newton(self.fkapa, 2.5, args=(deltahx, 1))
        self.w1h = (self.muhx - self.epsilon) / gamma(1 + 1 / self.kapah) + self.epsilon

    def transform(self, zk_col: np.ndarray):
      """
      Transforms a standard normal variable zk_col into the Weibull distribution using the sampling distribution h(x),
      and returns x, f(x), h(x) and the equivalent standard normal variable zf.
      """

      uk = norm.cdf(zk_col)  # Transform standard normal to uniform(0,1)

      # Sample Weibull (minimum) distribution with shift epsilon
      x = (self.w1h - self.epsilon) * (-np.log(1 - uk))**(1 / self.kapah) + self.epsilon

      fx = self.density_fx(x)
      hx = self.density_hx(x)

      cdfx = weibull_min.cdf((x - self.epsilon) / (self.w1f - self.epsilon), c=self.kapaf)
      zf = norm.ppf(cdfx)

      return x, fx, hx, zf
    
    def sample(self, ns: int):
      """
      Samples ns points from the sampling distribution h(x).
      """
      return weibull_min.rvs(
          c=self.kapah,
          loc=self.epsilon,
          scale=self.w1h - self.epsilon,
          size=ns
      )

    def density_fx(self, x: np.ndarray):
        """
        Computes the target density f(x) for given values of x.
        """
        y = (x - self.epsilon) / (self.w1f - self.epsilon)
        return weibull_min.pdf(y, c=self.kapaf) / (self.w1f - self.epsilon)

    def density_hx(self, x: np.ndarray):
        """
        Computes the sampling density h(x) for given values of x.
        """
        y = (x - self.epsilon) / (self.w1h - self.epsilon)
        return weibull_min.pdf(y, c=self.kapah) / (self.w1h - self.epsilon)

    def sample_direct(self, ns: int):
        """
        Generates samples from h(x) and returns x, f(x), h(x).
        """
        x = self.sample(ns)
        fx = self.density_fx(x)
        hx = self.density_hx(x)
        return x, fx, hx