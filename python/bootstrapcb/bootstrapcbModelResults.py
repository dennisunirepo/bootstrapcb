import numpy as np
import json
import time
import sys, os
from contextlib import contextmanager
from scipy.stats import norm, chi2, kstwobign
from scipy.optimize import minimize, NonlinearConstraint
from statsmodels.base.model import GenericLikelihoodModelResults
from statsmodels.tools.numdiff import approx_fprime, approx_hess

class bootstrapcbModelResults(GenericLikelihoodModelResults):
    def __init__(self, model, mlefit):
        """
        init results
        """
        super(bootstrapcbModelResults, self).__init__(model, mlefit)
        self.Params_bs = np.array([])
        self.Normalized_cov_params_bs = np.array([])

    def get_results(self):
        endog    = self.endog
        exog     = self.exog
        theta    = self.params                   
        cov      = self.normalized_cov_params    
        Theta_bs = self.Params_bs                
        Cov_bs   = self.Normalized_cov_params_bs 
        return endog, exog, theta, cov, Theta_bs, Cov_bs
        
    def get_results_as_dict(self):
        # get writeable data
        dict = {'endog':    self.endog.tolist(),
                'exog':     self.exog.tolist(), 
                'theta':    self.params.tolist(),
                'cov':      self.normalized_cov_params.tolist(),
                'Theta_bs': self.Params_bs.tolist(),
                'Cov_bs':   self.Normalized_cov_params_bs.tolist()}
        return dict
        
    def get_results_as_json(self):
        # get writeable data
        dict = self.get_results_as_dict()
        data = json.dumps(dict)
        return data

    def save_results_to_file(self, path):
        # get writeable data
        data = self.get_results_as_json()
        # write file
        file = open(path, "w")
        file.write(data)
        file.close() 
        
    def set_results_from_file(self, path):
        # get data
        file = open(path, "r")
        data = file.read()
        file.close() 
        # set results
        self.set_results_from_json(data)

    def set_results(self, endog, exog, theta, cov, Theta_bs, Cov_bs):
        self.model.endog              = endog
        self.model.exog               = exog
        self.params                   = theta
        self.normalized_cov_params    = cov
        self.Params_bs                = Theta_bs
        self.Normalized_cov_params_bs = Cov_bs
    
    def set_results_from_dict(self, dict):
        endog     = np.array(dict['endog'])
        exog      = np.array(dict['exog'])
        theta     = np.array(dict['theta'])
        cov       = np.array(dict['cov'])
        Theta_bs  = np.array(dict['Theta_bs'])
        Cov_bs  = np.array(dict['Cov_bs'])
        # set results
        self.set_results(endog, exog, theta, cov, Theta_bs, Cov_bs)

    def set_results_from_json(self, data):
        # parse json
        dict      = json.loads(data)
        self.set_results_from_dict(dict)

    def bootstrap_sample(self, nrep, parametric=False):
        """
        bootstrap method for the sample distribution
        """
        Endog_bs = []
        Exog_bs = []
        for _ in range(nrep):
            # generate bootstrap sample
            if parametric:
                tmp_data = self.model.rvs(*self.params, size=self.nobs)
                Endog_bs.append(tmp_data)
                Exog_bs.append(np.zeros_like(tmp_data))
            else:
                rvsind = np.random.randint(self.nobs, size=self.nobs)
                Endog_bs.append(self.model.endog[rvsind])
                Exog_bs.append(self.model.exog[rvsind])
        return np.array(Endog_bs), np.array(Exog_bs)

    @contextmanager
    def suppress_stdout(self):
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:  
                yield
            finally:
                sys.stdout = old_stdout

    def bootstrap_mle(self, nrep, parametric=False, bias_corrected=True):
        """
        bootstrap method for the mle distribution
        """
        Endog_bs, Exog_bs = self.bootstrap_sample(nrep=nrep, parametric=parametric)
        Theta_bs  = []
        Cov_bs    = []
        last_10_elapsed_times = []
        for i, _ in enumerate(Endog_bs):
            t = time.process_time()
            ####################################################
            with self.suppress_stdout():
                # initialize model with sampled data
                tmp_model = self.model.__class__(Endog_bs[i], Exog_bs[i], start_params=self.model.start_params) 
                # MLE estimation 
                tmp_results = tmp_model.fit()
                Theta_bs.append(tmp_results.params)
                if tmp_results.normalized_cov_params is None:
                    Cov_bs.append(np.nan*np.ones( (len(self.params),len(self.params)) ))
                else:
                    Cov_bs.append(tmp_results.cov_params()) 
            ####################################################
            # estimat time to go
            elapsed_time = time.process_time() - t
            last_10_elapsed_times.append(elapsed_time)
            if i >= 10: last_10_elapsed_times.pop(0)
            min_to_go = np.round( sum(last_10_elapsed_times)*(nrep - i)/
                                  (60*len(last_10_elapsed_times) ), 2)
            print('now at {}/{} - {}min to go - {}s per iteration'.format(i, 
                   nrep, min_to_go, elapsed_time))
        Theta_bs = np.array(Theta_bs, dtype=np.float)
        Cov_bs = np.array(Cov_bs, dtype=np.float)
        if bias_corrected:
            bias = np.array([np.mean(Theta_bs[:,i]) for i,_ in enumerate(self.params)]) - self.params
            Theta_bs = Theta_bs - bias
        # save bootstrapped mle distribution in class data
        self.Params_bs = Theta_bs
        self.Normalized_cov_params_bs = Cov_bs
        return Theta_bs, Cov_bs
    
    def likelihood_ratio_statistic(self, Theta): 
        """
        likelihood ratio statistic
        """
        # handle different shapes of Theta to enable meshgrid calculation
        shape = np.shape(Theta)
        if len(shape)>3: exit(0) #TODO
        elif len(shape)==1: 
            tmp_shape = (1, shape[0])
            final_shape = (1,)
        elif len(shape)==2:
            tmp_shape = (shape[0], shape[1])
            final_shape = (shape[0],)
        elif len(shape)==3:
            tmp_shape = (shape[0]*shape[1], shape[2]) 
            final_shape = (shape[0], shape[1]) 
        # calc lr statistic
        t = self.params
        chi2_vals = []
        for theta in np.reshape(Theta, tmp_shape):
            c = 2*(self.model.loglike(t) - self.model.loglike(theta))
            chi2_vals.append(c)
        return np.reshape(np.array(chi2_vals), final_shape)  
    
    def wald_statistic(self, Theta, Cov=None):
        """
        wald test statistic
        """
        # handle different shapes of Theta to enable meshgrid calculation
        shape = np.shape(Theta)
        if len(shape)>3: exit(0) #TODO
        elif len(shape)==1: 
            tmp_shape = (1, shape[0])
            final_shape = (1,)
        elif len(shape)==2:
            tmp_shape = (shape[0], shape[1])
            final_shape = (shape[0],)
        elif len(shape)==3:
            tmp_shape = (shape[0]*shape[1], shape[2]) 
            final_shape = (shape[0], shape[1]) 
        # calc wald statistic
        if Cov is None:
            cov = self.cov_params()
            cov_inv = np.linalg.inv(cov)
        t = self.params
        chi2_vals = []
        for i, theta in enumerate(np.reshape(Theta, tmp_shape)): 
            if Cov is not None:
                # use corresponding cov instead if present in the parameters
                cov = Cov[i]
                cov_inv = np.linalg.inv(cov)
            diff = np.subtract(theta, t)
            c = diff.T @ cov_inv @ diff
            chi2_vals.append(c)
        return np.reshape(np.array(chi2_vals), final_shape)   

    def percentile(self, T, alpha):
        """
        percentile method: alpha is in our case always the probability 
            level with probability 100(1 - alpha)%
        """ 
        return np.quantile(T, np.subtract(1, alpha))

    def mle(self, x):
        """
        returns fitted distplay_function
        """ 
        return self.model.display_func(x, *self.params)
  
    def confint(self, x, alpha, nrep=1000, parametric=True, method='nm'):
        """
        confidence interval
            methods: to construct pointwise confband
                delta - based on theorie
                bs    - bootstrapped confint for each x
                bst   - bootstrapped confint for each x per bootstrap-t
        """   
        # basic confidence interval (requires no bootstrap)
        if method=='delta':
            return self._confint_per_delta_method(x, alpha)

        # if no samples are given bootstrap
        if (len(self.Params_bs) == 0): 
            # sample from distribution 
            self.bootstrap_mle(nrep=nrep, parametric=parametric)

        # construct confidence interval
        if method=='bs' or method=='bst':
            return self._confint_per_bs(x, alpha, method=method)

    def confband(self, x, alpha, test='w', method='nm',
                 nrep=1000, parametric=True, 
                 T=None, Theta=None, df=None):
        """
        confidence band
            tests: for obtaining quantile
                asymp    - chi2 quantile
                w        - bootstrapped quantile per wald test 
                studw    - bootstrapped quantile per studentized wald test
                lr       - bootstrapped quantile per likelihood ratio test
                ''       - percentace quantile from some statistik T
            methods: to construct simultaneous confband
                bonf     - bonferroni cb
                bonf_bs  - bootstraped bonferroni cb
                ks       - based on ks-test
                ks_bs    - based on bootstrapped ks-test
                delta    - based on delta method
                nm       - per nealder mead search with point transformation 
                lagrange - per lagrange multiplier restriction 
                mc       - per monte-carlo-estimation from uniform points on elipse
                data     - by min-/maximizing the function for some given parameters
        """
        # basic confidence bands (requires no bootstrap or statistik T < t)
        if method=='bonf':
            m = len(x)
            return self.confint(x, alpha=alpha/m, method='delta')
        elif method=='bonf_bs':
            m = len(x)
            return self.confint(x, alpha=alpha/m, method='bst')
        elif method=='ks':
            m = len(self.model.endog)
            h = kstwobign.ppf(1 - alpha, loc=-0.5)/np.sqrt(m) # TODO somehow this is shifted
            return np.array([self.mle(x) - h, self.mle(x) + h]).T
        elif method=='ks_bs':
            m = len(self.model.endog)
            H_bs = []
            y = self.model.display_func(x, *self.params)
            for _, theta_bs in enumerate(self.Params_bs):
                y_bs = self.model.display_func(x, *theta_bs)
                H_bs.append( np.max(np.abs(y - y_bs)) )
            H_bs = np.array(H_bs)
            h = self.percentile(H_bs, alpha)
            return np.array([self.mle(x) - h, self.mle(x) + h]).T

        # ensure T will be used   
        if (T is not None): test = None
        # if no samples are given bootstrap
        if( (test != 'asymp')  and (test is not None) and (len(self.Params_bs) == 0) or 
            (test == 'studw')  and (len(self.Normalized_cov_params_bs) == 0) ):
            # sample from distribution 
            self.bootstrap_mle(nrep=nrep, parametric=parametric)

        # calculate critval t of the test statistic 
        if test=='asymp': # quantile from asymptotic theorie
            df = len(self.params) if df is None else df
            t = chi2.ppf(1 - alpha, df=df)
        elif test=='w': # wald test
            T_bs = self.wald_statistic(self.Params_bs)
            t    = self.percentile(T_bs, alpha=alpha) 
        elif test=='studw': # studentized wald test
            T_bs = self.wald_statistic(self.Params_bs, self.Normalized_cov_params_bs)
            t    = self.percentile(T_bs, alpha=alpha) 
        elif test=='lr': # likelihoodratio test
            T_bs = self.likelihood_ratio_statistic(self.Params_bs)
            t    = self.percentile(T_bs, alpha=alpha)
        else: # use statistic given in parameters
            t = self.percentile(T, alpha=alpha)  
        
        # construct confidence band from confregion
        if method=='delta':
            return self._confband_per_delta_method(x, c=t)
        elif method=='nm': # using optimization method to find min/max
            return self._confband_per_nm_search_on_confregion(x, level=t)
        elif method=='lagrange': # using optimization method to find min/max
            return self._confband_per_lagrange_multiplier(x, test=test, level=t)
        elif method=='mc': # uniform points on theoretical asymptotic sphere 
            Theta_in_confregion = self._params_uniform_on_confregion(t, nsamples=100)
            return self._confband_per_data(x, params=Theta_in_confregion)
        elif method=='data': # just by data given in parameters
            return self._confband_per_data(x, params=Theta)

    def _params_uniform_on_confregion(self, c, nsamples):
        """
        approximates the boundary of a confidence region based on asymptotic theorie (only points 
        that have property c = (mle - theta)^T * V^-1 * (mle - theta) are obtained)
        """
        cov = self.cov_params()
        chol = np.linalg.cholesky(cov)
        approx_cr = []
        for _ in range(nsamples):
            # obtain point uniform distributed on sphere
            z = norm.rvs(size=len(self.params), loc=0, scale=1)
            z = np.sqrt(c) * z / np.sqrt(np.sum(z ** 2))
            # transform to a point on the confidence region
            rv_mvn = chol @ z
            sample = np.add(rv_mvn, self.params)
            approx_cr.append(sample)
        return np.array(approx_cr)

    def _confband_per_data(self, x, params):
        """
        constructs a confidence band for a confidence region by min- maximizing the display
        function for all given params
        """
        # min- and maximize display_func for params ...
        y_l, y_u = np.array([np.inf] * len(x)), np.array([-np.inf] * len(x))
        for p in params:
            _display_func = self.model.display_func(x, *p)
            y_l = np.minimum(y_l, _display_func)
            y_u = np.maximum(y_u, _display_func)
        return np.array([y_l, y_u]).T

    def _confband_per_nm_search_on_confregion(self, x, level):
        """
        constructs a confidence band for a confidence region by min- maximizing the display
        function for all params on its edge
        """
        confband = np.zeros((len(x), 2))
        # min- and maximize display_func using optimization method
        cov = self.cov_params()
        chol = np.linalg.cholesky(cov)
        for i,_ in enumerate(x):
            # define functions to min-/maximize
            def f(theta_polar):
                if len(self.params) == 1:
                    # one parameter case
                    theta_cart = self.params + np.sqrt(level)*chol[0]*np.sin(theta_polar[0])
                else:
                    # point transformation from polar to cartesian
                    n = len(theta_polar)+1
                    theta_cart = np.zeros(n)
                    for k,_ in enumerate(theta_polar):
                        theta_cart[k] = np.sqrt(level) * np.prod([np.sin(theta_polar[j]) for j in range(k)]) * np.cos(theta_polar[k])
                    theta_cart[-1] = np.sqrt(level) * np.prod([np.sin(theta_polar[j]) for j in range(n-1)])
                    theta_cart = np.add(chol @ theta_cart, self.params)
                # function value at point on elipse
                return self.model.display_func(x[i], *theta_cart)
            def nf(theta_polar):
                return -f(theta_polar)
            x0_polar = np.zeros(len(self.params) - 1) if len(self.params) > 1 else 0
            res_l = minimize(f, x0_polar, method='nelder-mead',
                             options={'disp': True, 'fatol': 1e-3}) #, 'maxiter': 10000})
            res_u = minimize(nf, x0_polar, method='nelder-mead',
                             options={'disp': True, 'fatol': 1e-3}) #, 'maxiter': 10000})
            y_l = res_l.fun if res_l.success else np.nan
            y_u = -res_u.fun if res_u.success else np.nan
            confband[i] = [y_l, y_u]         
        return confband

    def _confband_per_lagrange_multiplier(self, x, test, level):
        """
        constructs a confidence band for a confidence region by min- maximizing the display
        function for all params inside
        """
        confband = np.zeros((len(x), 2))
        # get bounds and constraints 
        if test=='lr' or test=='asymp':
            cons_f = self.likelihood_ratio_statistic
        elif test=='w' or test=='studw': #use rather nm in this case
            cons_f = self.wald_statistic
        cov = self.cov_params()
        chol = np.linalg.cholesky(cov)
        constrain = NonlinearConstraint(cons_f, 0, level)
        for i,_ in enumerate(x):
            # define functions to min-/maximize
            f = lambda theta: self.model.display_func(x[i], *theta)
            nf = lambda theta: -self.model.display_func(x[i], *theta)
            while True:
                z = norm.rvs(size=len(self.params))
                theta_0 = np.add(chol @ ( np.sqrt(level) * z / np.sqrt(np.sum(z ** 2)) ), self.params) 
                try:
                    res_l = minimize(f, theta_0, method='trust-constr', #bounds=bounds,
                                        constraints=constrain, options={'verbose': 1, 'xtol': 1e-3,})
                    if res_l.success:
                        y_l = res_l.fun
                        break
                    break
                except Exception:
                    continue
            while True:
                z = norm.rvs(size=len(self.params))
                theta_0 = np.add(chol @ ( np.sqrt(level) * z / np.sqrt(np.sum(z ** 2)) ), self.params) 
                try:
                    res_u = minimize(nf, theta_0, method='trust-constr', #bounds=bounds,
                                        constraints=constrain, options={'verbose': 1, 'xtol': 1e-3})
                    if res_u.success:
                        y_u = -res_u.fun
                        break
                    break
                except Exception:
                    continue
            confband[i] = [y_l, y_u]               
        return confband

    def _confband_per_delta_method(self, x, c):
        """
        confidence band based on asymptotic theorie using taylor approximation
        """
        # construct confidence band from asymptotic theorie
        cov = self.cov_params()
        confband = np.zeros((len(x), 2))
        for i, _ in enumerate(x):
            y = self.model.display_func(x[i], *self.params)
            dy = approx_fprime(self.params, lambda t: self.model.display_func(x[i], *t), centered=True)
            if dy.size == 1:
                h = np.sqrt(c * dy * cov * dy)
            else:
                h = np.sqrt(c * dy.T @ cov @ dy)
            confband[i] = [y - h, y + h]
        return confband

    def _confint_per_delta_method(self, x, alpha):
        """
        confidence interval based on asymptotic theorie using taylor approximation
        """
        # get critval of standard norm for alpha
        z_alpha = norm.ppf(1 - alpha/2)
        # construct pointwise confidence band (or ints) from asymptotic theorie
        cov = self.cov_params()
        confint = np.zeros((len(x), 2))
        for i, _ in enumerate(x):
            y = self.model.display_func(x[i], *self.params)
            dy = approx_fprime(self.params, lambda t: self.model.display_func(x[i], *t), centered=True)
            if dy.size == 1:
                h = z_alpha * np.sqrt(dy * cov * dy)
            else:
                h = z_alpha * np.sqrt(dy.T @ cov @ dy)
            confint[i] = [y - h, y + h]
        return confint

    def _confint_per_bs(self, x, alpha, method):
        """
        confidence intervals bootstrapped in each x_i 
        """
        # construct pointwise confidence band (or ints) 
        n = len(self.Params_bs)
        confint = np.zeros((len(x), 2))
        if method=='bs':
            for i, _ in enumerate(x):
                y = self.model.display_func(x[i], *self.params)
                Y_bs = [y - self.model.display_func(x[i], *theta_bs) for theta_bs in self.Params_bs]
                # percentiles
                y_l = y + self.percentile(Y_bs, 1 - alpha) 
                y_u = y + self.percentile(Y_bs, alpha)
                confint[i] = [y_l, y_u] 
        elif method=='bst': # studentized ci 
            # TODO broken
            for i, _ in enumerate(x):
                y = self.model.display_func(x[i], *self.params)
                Y_bs = [self.model.display_func(x[i], *theta_bs) for theta_bs in self.Params_bs]
                v = np.sum( (y - Y_bs)**2 )/(n - 1) 
                # variance per mle based on delta method
                V_bs = []
                for j, _ in enumerate(self.Params_bs):
                    y_bs = self.model.display_func(x[i], *self.Params_bs[j])
                    dy_bs = approx_fprime(self.Params_bs[j], lambda t: self.model.display_func(x[i], *t), centered=True)
                    # studentized version of y_bs
                    if dy_bs.size == 1:
                        v_bs = dy_bs * self.Normalized_cov_params_bs[j] * dy_bs
                    else:
                        v_bs = dy_bs.T @ self.Normalized_cov_params_bs[j] @ dy_bs
                    V_bs.append(v_bs)
                # studentized version of Y_bs
                Y_bst = (y - Y_bs)/np.sqrt(V_bs)
                # studentized percentiles
                y_l = y + self.percentile(Y_bst, 1 - alpha) * np.sqrt(v) # t.ppf(alpha, df=n-1)
                y_u = y + self.percentile(Y_bst, alpha) * np.sqrt(v) # t.ppf(1 - alpha, df=n-1)
                confint[i] = [y_l, y_u]               
        return confint
