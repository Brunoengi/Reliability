from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 

class Uniform(AbstractDistribution):
  def __init__(self, dictionaryInfo):

    ValidateDictionary.check_possible_arrays_keys(dictionaryInfo, ['varmean','varstd'], ['parameter1', 'parameter2'])
    
    if(set(['parameter1','parameter2']).issubset(dictionaryInfo)):
      a = dictionaryInfo['parameter1']
      b = dictionaryInfo['parameter2']
    
      self.varmean = float((a + b) / 2)
      self.varstd = float((b - a) / np.sqrt(12))
      self.varhmean = float(self.varmean)

    super().__init__(dictionaryInfo)
    ValidateClass.has_invalid_key(self, 'varname', 'vardist', 'varmean', 'varcov', 'varstd', 'varhmean', 'parameter1', 'parameter2')