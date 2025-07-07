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

        self.loc = self.a
        self.scale = self.b - self.a

        # Target distribution parameters (f(x))
        self.varmean = self.a + (self.q / (self.q + self.r)) * (self.b - self.a)
        self.varstd = np.sqrt((self.q * self.r) / (((self.q + self.r) ** 2) * (self.q + self.r + 1)) * (self.b - self.a) ** 2)

        self.mufx = self.varmean
        self.sigmafx = self.varstd

        super().__init__(props)
        self.update_sampling(nsigma=1.0)

    def validate_specific_parameters(self, props):
        ValidateDictionary.is_dictionary(props)
        ValidateDictionary.check_possible_arrays_keys(props, ['parameter1', 'parameter2', 'parameter3', 'parameter4'])

    def beta_limits(self, vars, mux, sigmax, q, r):
        """System of equations to solve for a and b of a Beta distribution with given mean and std."""
        a_, b_ = vars
        eq1 = a_ + (q / (q + r)) * (b_ - a_) - mux
        eq2 = np.sqrt((q * r) / (((q + r) ** 2) * (q + r + 1))) * (b_ - a_) - sigmax
        return [eq1, eq2]

    def update_sampling(self, nsigma: float = 1.0):
      """
      Update h(x) parameters using scaled std deviation.
      """
      self.muhx = self.varhmean
      self.sigmahx = self.sigmafx * nsigma
      self.ah, self.bh = fsolve(self.beta_limits, (self.a, self.b), args=(self.muhx, self.sigmahx, self.q, self.r))
      self.loch = self.ah
      self.scaleh = self.bh - self.ah

    def sample(self, ns: int):
        """
        Generate ns samples from the importance distribution h(x).
        """
        return beta_dist.rvs(self.q, self.r, loc=self.ah, scale=self.scaleh, size=ns)

    def density_fx(self, x: np.ndarray):
        """
        Compute the PDF of the target distribution f(x).
        """
        return beta_dist.pdf(x, self.q, self.r, loc=self.loc, scale=self.scale)

    def density_hx(self, x: np.ndarray):
        """
        Compute the PDF of the sampling distribution h(x).
        """
        return beta_dist.pdf(x, self.q, self.r, loc=self.loch, scale=self.scaleh)

    def transform(self, zk_col: np.ndarray):
      """
      Transforma variáveis normais padrão zk_col em amostras x da distribuição h(x).
      """
      uk = norm.cdf(zk_col)
      loc_hx = self.ah
      scale_hx = self.bh - self.ah
      x = beta_dist.ppf(uk, self.q, self.r, loc=loc_hx, scale=scale_hx)

      fx = self.density_fx(x)
      hx = self.density_hx(x)

      cdfx = beta_dist.cdf(x, self.q, self.r, loc=self.a, scale=self.b - self.a)
      cdfx = np.clip(cdfx, 1e-10, 1.0 - 1e-10)
      zf = norm.ppf(cdfx)

      return x, fx, hx, zf

    def sample_direct(self, ns: int):
        """
        Generate ns samples from f(x) and compute their densities under f(x) and h(x).
        """
        x = beta_dist.rvs(self.q, self.r, self.loch, self.scaleh, size=ns)
        fx = self.density_fx(x)
        hx = self.density_hx(x)
        return x, fx, hx