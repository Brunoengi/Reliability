from distribution.AbstractDistribution import AbstractDistribution 
from utils.validate.base_types.validate_dictionary import ValidateDictionary
from utils.validate.base_types.validate_class import ValidateClass 
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm, uniform

class Uniform(AbstractDistribution):
    def __init__(self, props):
        if {'parameter1', 'parameter2'}.issubset(props):
            self.a = props['parameter1']
            self.b = props['parameter2']
            self.varmean = float((self.a + self.b) / 2)
            self.varstd = float((self.b - self.a) / np.sqrt(12))
        elif {'varmean', 'varstd'}.issubset(props):
            self.varmean = float(props['varmean'])
            self.varstd = float(props['varstd'])
            self.a, self.b = fsolve(self.uniform_limits, (self.varmean - 1, self.varmean + 1), args=(self.varmean, self.varstd))
        else:
            raise ValueError("Uniform distribution requires either ['parameter1','parameter2'] or ['varmean','varstd'].")

        self.mufx = self.varmean
        self.sigmafx = self.varstd

        props.update({
            'varmean': self.varmean,
            'varstd': self.varstd,
            'parameter1': self.a,
            'parameter2': self.b
        })

        super().__init__(props)

        # Initialize sampling distribution parameters
        self.update_sampling(nsigma=1.0)

    def validate_specific_parameters(self, props):
        ValidateDictionary.check_possible_arrays_keys(props, ['varmean','varstd'], ['parameter1', 'parameter2'])

    @staticmethod
    def uniform_limits(vars, mux, sigmax):
        a_, b_ = vars
        eq1 = (a_ + b_) / 2 - mux
        eq2 = (b_ - a_) / np.sqrt(12) - sigmax
        return [eq1, eq2]

    def update_sampling(self, nsigma: float = 1.0):
        """Update the parameters of the sampling distribution h(x)."""
        self.muhx = self.mufx
        self.sigmahx = self.sigmafx * nsigma
        self.ah, self.bh = fsolve(self.uniform_limits, (self.a, self.b), args=(self.muhx, self.sigmahx))

    def transform(self, zk_col: np.ndarray):
        if not hasattr(self, 'ah') or not hasattr(self, 'bh'):
            raise RuntimeError("Sampling distribution parameters not defined. Call update_sampling(nsigma) first.")

        uk = norm.cdf(zk_col)
        x = self.ah + (self.bh - self.ah) * uk
        zf = norm.ppf(uniform.cdf(x, loc=self.a, scale=self.b - self.a))

        fx = self.density_fx(x)
        hx = self.density_hx(x)
        
        return x, fx, hx, zf
    
    def sample(self, ns: int):
        """
        Sample x from the sampling distribution h(x).
        """
        return uniform.rvs(loc=self.ah, scale=self.bh - self.ah, size=ns)

    def density_fx(self, x: np.ndarray):
        """
        Evaluate the PDF of the target distribution f(x).
        """
        return uniform.pdf(x, loc=self.a, scale=self.b - self.a)
    
    def density_hx(self, x: np.ndarray):
        """
        Evaluate the PDF of the sampling distribution h(x).
        """
        return uniform.pdf(x, loc=self.ah, scale=self.bh - self.ah)
    
    def sample_direct(self, ns: int):
        """
        Sample x from h(x), and return f(x), h(x) evaluated at x.
        This is useful for Monte Carlo sampling without transformation.
        """
        x = self.sample(ns)
        fx = self.density_fx(x)
        hx = self.density_hx(x)
        return x, fx, hx