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
        uk = norm.cdf(zk_col)  # z ~ N(0,1) -> U(0,1)
        x = self.vhn / (np.log(1 / uk)) ** (1 / self.kapah)

        ynf = x / self.vfn
        ynh = x / self.vhn

        cdfx = invweibull.cdf(ynf, self.kapaf)
        zf = norm.ppf(cdfx)

        fx = invweibull.pdf(ynf, self.kapaf) / self.vfn
        hx = invweibull.pdf(ynh, self.kapah) / self.vhn

        return x, fx, hx, zf