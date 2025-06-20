from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np

class Beta(AbstractDistribution):
  def __init__(self, dictionaryInfo):

    ValidateDictionary.is_dictionary(dictionaryInfo)
    ValidateDictionary.check_possible_arrays_keys(dictionaryInfo, ['parameter1', 'parameter2', 'parameter3', 'parameter4'])

    a = dictionaryInfo['parameter1']
    b = dictionaryInfo['parameter2']
    q = dictionaryInfo['parameter3']
    r = dictionaryInfo['parameter4']

    self.varmean = float(a + q / (q + r) * (b - a))
    self.varstd = float(np.sqrt((q * r) / ((q + r) **2 * (q + r + 1)) * (b - a) ** 2))
    self.varhmean = float(self.varmean)

    super().__init__(dictionaryInfo)
    ValidateClass.has_invalid_key(self, 'varname', 'vardist','varmean','varcov','varstd','varhmean','parameter1','parameter2','parameter3','parameter4')