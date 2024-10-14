from src.distribution.AbstractDistribution import AbstractDistribution 
from src.utils.validate.Dictionary import ValidateDictionary
from src.utils.validate.Class import ValidateClass

class Gumbel(AbstractDistribution):
  def __init__(self, dictionaryInfo):

    ValidateDictionary.is_dictionary(dictionaryInfo)
    ValidateDictionary.check_possible_arrays_keys(dictionaryInfo, ['varmean', 'varstd'], ['varmean', 'varcov'])
    super().__init__(dictionaryInfo)
    ValidateClass.has_invalid_key(self, 'varname', 'vardist', 'varmean', 'varstd', 'varcov', 'varhmean')