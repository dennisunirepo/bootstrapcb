import numpy as np
from scipy.stats import gamma, rv_continuous
from bootstrapcb import bootstrapcbModel, genericAnalysis

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

# init analysis
ia, s = 1, 0.8
iar, sr = 1/ia, 1/s
shape, scale = 1, 1/(sr - iar)
theta_0 = [shape*scale, np.sqrt(shape)*scale] 
path = "datasets/"
path = "D:/Dennis/Desktop/BA/unirepoabgabe/omnetpp/example_mg1/simulations/datasets/"
sim_runs_dataset = "analysis_W.csv"
bootstrapcb_dataset = "simstudy_W_gamma_mu_sigma.json"
simstudy_dataset = "simstudy_W_gamma_mu_sigma.txt"
results_test = genericAnalysis(path + sim_runs_dataset,
                               path + bootstrapcb_dataset, 
                               path + simstudy_dataset,
                               path + 'fig/',
                               model_class=model, 
                               theta_0=theta_0)

# get run dataset
# statistic = "waitingTime"
# strategy = "batch_values"
# nbatches = 500
# results_test.load_data_from_sim_dataset(statistic=statistic, strategy=strategy, nbatches=nbatches)
########################################################################
# results_test.data_per_run = results_test.data_per_run[:100]
# results_test.data_per_run[0]['endog'] = results_test.data_per_run[0]['endog'][:30]

# get bootstrap dataset
# nrep = 1000 # per expirience bs is accurate at 1000 reps
# parametric = True
# results_test.generate_bs_dataset(nrep=nrep, parametric=parametric)
results_test.init_dataset(show_first_run=False)
########################################################################
results_test.fitresults_per_run = results_test.fitresults_per_run[:100]

### show
results_test.get_information_for_dataset()
# results_test.show_dataset_at_run(0, show_seperated=True, save_figures=False) 
######################################################################## 
# results_test.results.set_results_from_dict(results_test.fitresults_per_run[0]) 
# results_test.data_plots(nx=100, xmin=0, xmax=3)
# results_test.confregion_plots()
# plt.show()
# results_test.confband_plots(nx=50, xmin=0, xmax=3)
# plt.show()

### simstudy
xs = np.linspace(2, 12, 10)
levels = [0.5,0.1,0.05,0.01]
tests = ['w', 'w_bs', 'studw_bs', 'lr', 'lr_bs']
configs = [{
    'method': 'bonf',
    'test':   ''
},{
    'method': 'ks',
    'test':   ''
},{
    'method': 'ks_bs',
    'test':   ''
},{
    'method': 'delta',
    'test':   'asymp'
},{
    'method': 'mc',
    'test':   'asymp'
},{
    'method': 'mc',
    'test':   'w'
},{
    'method': 'mc',
    'test':   'studw'
},{
    'method': 'nm',
    'test':   'asymp'
},{
    'method': 'nm',
    'test':   'w'
},{
    'method': 'nm',
    'test':   'studw'
},{
    'method': 'lagrange',
    'test':   'asymp'
},{
    'method': 'lagrange',
    'test':   'lr'
}]
results_test.simstudy(xs=xs, levels=levels, tests=tests, configs=configs, 
                      which='both', show=True)