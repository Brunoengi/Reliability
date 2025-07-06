from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np

class Uniform(AbstractDistribution):
  def __init__(self, props):

    if(set(['parameter1','parameter2']).issubset(props)):
      a = props['parameter1']
      b = props['parameter2']
    
      self.varmean = float((a + b) / 2)
      self.varstd = float((b - a) / np.sqrt(12))
      self.varhmean = float(self.varmean)

    super().__init__(props)

  def validate_specific_parameters(self, props):
    ValidateDictionary.check_possible_arrays_keys(props, ['varmean','varstd'], ['parameter1', 'parameter2'])