from abc import ABC, abstractmethod

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
