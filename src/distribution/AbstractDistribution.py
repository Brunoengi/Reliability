from abc import ABC, abstractmethod
from scipy.stats import norm
import numpy as np

from utils.validate.domain_types.validate_xvar import ValidateXvar

class AbstractDistribution(ABC):
  def __init__ (self, props: dict):
      
    ValidateXvar(props)
  
    self.set_properties(props)
    self.set_initial_values()

    self.namedist = self.vardist.lower()

    self.sigmafx = float(self.varstd)
    self.muhx = float(self.varhmean)
    self.mufx = float(props['varmean'])

  @abstractmethod
  def validate_specific_parameters(self, props):
      pass

  def set_properties(self, props):
    for key, value in props.items():
      setattr(self, key, value)
  
  def set_initial_values(self):
      # Checks whether 'varhmean' was provided; if not, use 'varmean'
      self.varhmean = float(getattr(self, 'varhmean', self.varmean))
  
      # Tests whether 'varstd' exists as an attribute and calculates 'varcov' if possible
      if hasattr(self, 'varstd'):
        self.varcov = float(self.varstd / self.varmean) if self.varmean > 0 else 1.00
      else:
        self.varstd = float(self.varcov * self.varmean)
  
  def update_weights(
        self,
        fx: np.ndarray,
        hx: np.ndarray,
        zf_col: np.ndarray,
        zk_col: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates importance sampling weights for this distribution.

        Parameters:
            fx: PDF values of target distribution at samples x.
            hx: PDF values of sampling distribution at samples x.
            zf_col: Standard normal transformed samples for target distribution.
            zk_col: Standard normal transformed samples for sampling distribution.

        Returns:
            w: importance sampling weights (fx/hx adjusted by normal PDFs).
            fx_over_phi_zf: intermediate term fx / phi(zf) useful for weight updates.
        """
        phi_zf = norm.pdf(zf_col)
        phi_zk = norm.pdf(zk_col)

        w = (fx / phi_zf) / (hx / phi_zk)
        fx_over_phi_zf = fx / phi_zf
        return w, fx_over_phi_zf
