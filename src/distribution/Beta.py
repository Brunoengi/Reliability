from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np

class Beta(AbstractDistribution):
  def __init__(self, props):

    self.validate_specific_parameters(props)

    self.a = props['parameter1']
    self.b = props['parameter2']
    self.q = props['parameter3']
    self.r = props['parameter4']

    self.varmean = float(self.a + self.q / (self.q + self.r) * (self.b - self.a))
    self.varstd = float(np.sqrt((self.q * self.r) / ((self.q + self.r) **2 * (self.q + self.r + 1)) * (self.b - self.a) ** 2))
    self.varhmean = float(self.varmean)

    super().__init__(props)
    

  def validate_specific_parameters(self, props):
    ValidateDictionary.is_dictionary(props)
    ValidateDictionary.check_possible_arrays_keys(props, ['parameter1', 'parameter2', 'parameter3', 'parameter4'])