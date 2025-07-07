# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:32:15 2024

@author: BrunoTeixeira
"""

from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_class import ValidateClass
from utils.validate.base_types.validate_dictionary import ValidateDictionary 
import numpy as np
from scipy.stats import norm

class Normal(AbstractDistribution):
  def __init__(self, props: dict):
    
    self.validate_specific_parameters(props)
    super().__init__(props) 

    self.update_sampling(nsigma=1.0)


  def validate_specific_parameters(self, props):
    ValidateDictionary.is_dictionary(props)
    ValidateDictionary.has_keys(props,'varmean')
    ValidateDictionary.is_float(props, 'varmean')
    ValidateDictionary.check_keys_count(props, 1, 'varcov', 'varstd')
    ValidateDictionary.check_if_exists(props, 'varcov', lambda d, k: ValidateDictionary.is_greater_or_equal_than(d, k, 0))

  def update_sampling(self, nsigma: float = 1.0):
    """
    Update the parameters of the sampling distribution h(x) by scaling the
    standard deviation with nsigma. The mean of h(x) is set equal to the
    mean of the target distribution f(x).
    """
    self.muhx = self.mufx
    self.sigmahx = self.sigmafx * nsigma

  def sample(self, ns):
    """
    Generate ns random samples directly from the sampling distribution h(x),
    modeled as a normal distribution with parameters (muhx, sigmahx).
    """
    return norm.rvs(loc=self.muhx, scale=self.sigmahx, size=ns)

  def density_fx(self, x):
    """
    Compute the probability density function (PDF) of the target distribution f(x)
    evaluated at samples x.
    """
    return norm.pdf(x, loc=self.mufx, scale=self.sigmafx)

  def density_hx(self, x):
    """
    Compute the PDF of the sampling distribution h(x) evaluated at samples x.
    """
    return norm.pdf(x, loc=self.muhx, scale=self.sigmahx)

  def transform(self, zk_col):
    """
    Transform standard normal variables zk_col into samples x of the sampling
    distribution h(x), and calculate the PDFs of f(x) and h(x) at x. Also returns
    the corresponding standard normal values zf for the target distribution.
    """
    x = self.muhx + self.sigmafx * zk_col
    fx = self.density_fx(x)
    hx = self.density_hx(x)
    zf = (x - self.mufx) / self.sigmafx
    return x, fx, hx, zf

  def sample_direct(self, ns):
    """
    Generate ns samples directly from h(x) and compute their PDFs under both
    the target distribution f(x) and the sampling distribution h(x).
    """
    x = self.sample(ns)
    fx = self.density_fx(x)
    hx = self.density_hx(x)
    return x, fx, hx
