import numpy as np
from scipy.stats import rv_discrete, rv_continuous
from statsmodels.base.model import GenericLikelihoodModel
from .bootstrapcbModelResults import bootstrapcbModelResults

class bootstrapcbModel(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, start_params=None, timestamps=None, **kwds):
        """
        Initialize likelihood and exog, endog data
        """                
        self.start_params = start_params
        self.timestamps = timestamps
        if not hasattr(self, 'dist'):
            self.dist = self.distribution(name='local_distribution')
        
        if exog is None:
            exog = np.zeros_like(endog)

        super(bootstrapcbModel, self).__init__(endog=endog, exog=exog, **kwds)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        """
        MLE estimation and returning own result-class
        """
        if start_params is None:
            if hasattr(self, 'start_params') and self.start_params is not None:
                start_params = self.start_params
            else:
                try:
                    start_params = self.dist.fit(self.endog)  # TODO
                except Exception:
                    pass
            
        fit_method = super(bootstrapcbModel, self).fit
        genericmlefit = fit_method(start_params=start_params,
                                   maxiter=maxiter, maxfun=maxfun, **kwds)
        # wrap results into the bootstrapcbModelResults class
        bscbmlefit = bootstrapcbModelResults(self, genericmlefit)
        return bscbmlefit

    def ecdf(self, x):
        return [len(self.endog[(self.endog <= x_i)])/len(self.endog) for x_i in x]

    def pdf(self, x, *args, **kwds):
        return self.dist.pdf(x, *args, **kwds)

    def logpdf(self, x, *args, **kwds):
        return self.dist.logpdf(x, *args, **kwds)

    def pmf(self, x, *args, **kwds):
        return self.dist.pmf(x, *args, **kwds)

    def logpmf(self, x, *args, **kwds):
        return self.dist.logpmf(x, *args, **kwds)

    def cdf(self, x, *args, **kwds):
        return self.dist.cdf(x, *args, **kwds)

    def logcdf(self, x, *args, **kwds):
        return self.dist.logcdf(x, *args, **kwds)

    def ppf(self, q, *args, **kwds):
        return self.dist.ppf(q, *args, **kwds)

    def rvs(self, *args, **kwds):
        return self.dist.rvs(*args, **kwds)
        
    def nloglikeobs(self, params): 
        x = self.endog  
        if isinstance(self.dist, rv_continuous):
            return -self.dist.logpdf(x, *params)
        elif isinstance(self.dist, rv_discrete):
            return -self.dist.logpmf(x, *params)
          
