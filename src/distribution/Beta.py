from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import beta as beta_dist, norm


class Beta(AbstractDistribution):
    def __init__(self, props):
        self.validate_specific_parameters(props)

        self.a = props['parameter1']
        self.b = props['parameter2']
        self.q = props['parameter3']
        self.r = props['parameter4']

        # Mean and standard deviation of the target distribution f(x)
        self.varmean = float(self.a + self.q / (self.q + self.r) * (self.b - self.a))
        self.varstd = float(np.sqrt((self.q * self.r) / ((self.q + self.r) ** 2 * (self.q + self.r + 1)) * (self.b - self.a) ** 2))

        self.mufx = self.varmean
        self.sigmafx = self.varstd

        # Initialize sampling distribution h(x) parameters
        super().__init__(props)
        self.update_sampling(nsigma=1.0)

    def validate_specific_parameters(self, props):
        ValidateDictionary.is_dictionary(props)
        ValidateDictionary.check_possible_arrays_keys(props, ['parameter1', 'parameter2', 'parameter3', 'parameter4'])

    def beta_limits(self, vars, mux, sigmax, q, r):
        """System of equations to solve for a and b based on mean and std."""
        a_, b_ = vars
        eq1 = a_ + q / (q + r) * (b_ - a_) - mux
        eq2 = np.sqrt((q * r) / (((q + r) ** 2) * (q + r + 1))) * (b_ - a_) - sigmax
        return [eq1, eq2]

    def update_sampling(self, nsigma: float = 1.0):
        """Update sampling distribution h(x) parameters based on nsigma."""
        self.muhx = self.mufx
        self.sigmahx = self.sigmafx * nsigma

        self.ah, self.bh = fsolve(self.beta_limits, (self.a, self.b), args=(self.muhx, self.sigmahx, self.q, self.r))

    def transform(self, zk_col: np.ndarray):
        
        uk = norm.cdf(zk_col)

        # Location and scale parameters for target and sampling distributions
        loc_fx, scale_fx = self.a, self.b - self.a
        loc_hx, scale_hx = self.ah, self.bh - self.ah

        # Transform standard normal to beta h(x)
        x = beta_dist.ppf(uk, self.q, self.r, loc=loc_hx, scale=scale_hx)

        # Compute z_f using the target distribution
        cdfx = beta_dist.cdf(x, self.q, self.r, loc=loc_fx, scale=scale_fx)
        zf = norm.ppf(cdfx)

        fx = beta_dist.pdf(x, self.q, self.r, loc=loc_fx, scale=scale_fx)
        hx = beta_dist.pdf(x, self.q, self.r, loc=loc_hx, scale=scale_hx)

        return x, fx, hx, zf