import numpy as np
from scipy.stats import gamma, rv_continuous
from bootstrapcb import bootstrapcbModel

from scipy.stats import chi2
import matplotlib.pyplot as plt
    
class exampleModel(bootstrapcbModel):
  # local distribution class
  class distribution(rv_continuous):     
    def _cdf(self, x, mu, sigma):
      a = mu**2 / sigma**2
      b = sigma**2 / mu
      return gamma.cdf(x, a, scale=b) 
    def _pdf(self, x, mu, sigma):
      a = mu**2 / sigma**2
      b = sigma**2 / mu
      return gamma.pdf(x, a, scale=b) 
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

  def loglike(self, theta): # for dist 
    x = self.endog
    mu = theta[0]
    sigma = theta[1]
    a = mu**2 / sigma**2
    b = sigma**2 / mu
    return -gamma.nnlf([a,0,b], x) 
    
  # function for which the confidence band is to be calculated
  def display_func(self, x, *args, **kwds):
    return self.cdf(x, *args, **kwds)

# generate some random data and fit our model
data = gamma.rvs(a=2, size=20)
myModle = exampleModel(data, start_params=[1,1])
myResults = myModle.fit()
print(myResults.summary())

###################################

def ecdf(x, X):
    return np.array([len(X[(X < x_i)])/len(X) for x_i in x])

# calculate bootstrap statistics
mle_bs, _ = myResults.bootstrap_mle(1000, parametric=True)
w_bs = myResults.wald_statistic(mle_bs)
lr_bs = myResults.likelihood_ratio_statistic(mle_bs)
xs = np.linspace(0, chi2.ppf(0.99, 2), 100)

plt.plot(xs, chi2.cdf(xs, 2), label='Chi2(2)')
plt.plot(xs, ecdf(xs, w_bs),  label='bs wald-statistic', color='green', 
                              drawstyle='steps-post')
plt.plot(xs, ecdf(xs, lr_bs), label='bs likelihood-ratio', color='cyan',  
                              drawstyle='steps-post')
plt.legend()
plt.show()

###################################

# meshgrid for 3d-wald-statistic
mle = myResults.params
cov = myResults.cov_params()
xmin = mle[0] - 5*np.sqrt(cov[0][0])
xmax = mle[0] + 5*np.sqrt(cov[0][0])
ymin = mle[1] - 5*np.sqrt(cov[1][1])
ymax = mle[1] + 5*np.sqrt(cov[1][1])
xs   = np.linspace(xmin, xmax, 50)
ys   = np.linspace(ymin, ymax, 50)
X, Y = np.meshgrid(xs, ys)
XY   = np.dstack((X, Y))
Z_w = myResults.wald_statistic(XY)
Z_lr = myResults.likelihood_ratio_statistic(XY)

# calculate conf_region and uniform distributed points on the elipse
chi2_levels = myResults.percentile(w_bs, alpha=[0.5,0.05,0.01])
uniform_on_confregion = myResults._params_uniform_on_confregion(chi2_levels[1], 
                                                                nsamples=1000)

plt.plot(mle_bs[:, 0], mle_bs[:, 1], 'o', alpha=0.1, label='bs mles')
plt.plot(uniform_on_confregion[:, 0], uniform_on_confregion[:, 1], 'o', 
         alpha=0.1, label='samples uniform on confregion')
plt.contour(X, Y, Z_w, levels=chi2_levels, colors='green', linestyles='--')
plt.contour(X, Y, Z_lr, levels=chi2_levels, colors='cyan', linestyles='--')
plt.legend()
plt.show()

###################################

# calculate conf_band
xs = np.linspace(myResults.model.ppf(0.01, *mle), 
                 myResults.model.ppf(0.99, *mle), 50)
fitted_func = myResults.mle(xs)
confband    = myResults.confband(xs, alpha=0.1, test='w', method='nm')

plt.plot(xs, ecdf(xs, data), label='empirical distribution', 
                             drawstyle='steps-post')
plt.plot(xs, fitted_func,    label='fitted distribution')
plt.plot(xs, confband[:, 0], label='confidence band',
                             color='green', linestyle='--')
plt.plot(xs, confband[:, 1], color='green', linestyle='--')
plt.legend()
plt.show()