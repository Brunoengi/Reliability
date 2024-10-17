from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.Dictionary import ValidateDictionary
from utils.validate.Class import ValidateClass

class Gama(AbstractDistribution):
  def __init__(self, dictionaryInfo):

    ValidateDictionary.is_dictionary(dictionaryInfo)
    ValidateDictionary.check_possible_arrays_keys(dictionaryInfo, ['varmean', 'varcov'])
    super().__init__(dictionaryInfo)
    ValidateClass.has_invalid_key(self, 'varname', 'vardist', 'varmean', 'varstd', 'varhmean')
    