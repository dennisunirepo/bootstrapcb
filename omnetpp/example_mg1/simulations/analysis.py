import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, rv_continuous
from bootstrapcb import bootstrapcbModel

class model(bootstrapcbModel):
    # local distribution class
    class distribution(rv_continuous):     
        def _cdf(self, x, mu, sigma):
            a = mu**2 / sigma**2
            b = sigma**2 / mu
            return gamma.cdf(x, a, scale=b) 
        def _logcdf(self, x, mu, sigma):
            a = mu**2 / sigma**2
            b = sigma**2 / mu
            return gamma.logcdf(x, a, scale=b) 
        def _pdf(self, x, mu, sigma):
            a = mu**2 / sigma**2
            b = sigma**2 / mu
            return gamma.pdf(x, a, scale=b) 
        def _logpdf(self, x, mu, sigma):
            a = mu**2 / sigma**2
            b = sigma**2 / mu
            return gamma.logpdf(x, a, scale=b) 
        def _ppf(self, q, mu, sigma):
            a = mu**2 / sigma**2
            b = sigma**2 / mu
            return gamma.ppf(q, a, scale=b) 
        def _rvs(self, mu, sigma, size=None, random_state=None):
            a = mu**2 / sigma**2
            b = sigma**2 / mu
            return gamma.rvs(a, scale=b, size=size, random_state=random_state)
        def _argcheck(self, mu, sigma):
            return sigma >= 0

    # function for which the confidence band is to be calculated
    def display_func(self, x, *args, **kwds):
        return self.cdf(x, *args, **kwds)

def parse_if_number(s):
    try: return float(s)
    except: return True if s=="true" else False if s=="false" else s if s else None

def parse_ndarray(s):
    return np.fromstring(s, sep=' ') if s else None

df = pd.read_csv("./datasets/analysis_W.csv", 
    converters = {
        'attrvalue': parse_if_number,
        'binedges': parse_ndarray,
        'binvalues': parse_ndarray,
        'vectime': parse_ndarray,
        'vecvalue': parse_ndarray
    })
print(df.head()) # print an excerpt of the result
df = df[(df.type=='vector') & (df.name=='waitingTime:vector')]
values = df.iloc[0]["vecvalue"][:50]
tmpModle = model(values, start_params=[1, 1/(1.25 - 1)])
tmpResults = tmpModle.fit()
tmpMle = tmpResults.params
print(tmpResults.summary())

xs = np.linspace(tmpResults.model.ppf(0.01, *tmpMle), 
tmpResults.model.ppf(0.99, *tmpMle), 50)
ecdf = tmpResults.model.ecdf(xs)
plt.plot(xs, ecdf, label='empirical distribution', 
                   drawstyle='steps-post')
fitted_func = tmpResults.mle(xs)
plt.plot(xs, fitted_func, label='fitted distribution')
confband = tmpResults.confband(xs, alpha=0.1, nrep=100, parametric=True,
                               test='w', method='nm')
plt.plot(xs, confband[:, 0], label='confidence band',
                             color='green', linestyle='--')
plt.plot(xs, confband[:, 1], color='green', linestyle='--')
plt.legend()
plt.show()