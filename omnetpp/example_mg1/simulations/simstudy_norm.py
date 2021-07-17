import numpy as np
from scipy.stats import norm
from bootstrapcb import bootstrapcbModel, genericAnalysis

class model(bootstrapcbModel):
    # local distribution class
    def __init__(self, endog, exog=None, start_params=None, timestamps=None):
        self.dist = norm
        super(model, self).__init__(endog=endog, exog=exog, start_params=start_params, timestamps=timestamps)

    # function for which the confidence band is to be calculated
    def display_func(self, x, *args, **kwds):
        return self.cdf(x, *args, **kwds)

# init analysis
mu, sigma = 0, 1
theta_0 = [mu, sigma] 
path = "datasets/"
sim_runs_dataset = "tmp.csv"
bootstrapcb_dataset = "simstudy_norm.json"
simstudy_dataset = "simstudy_norm.txt"
results_test = genericAnalysis(path + sim_runs_dataset,
                               path + bootstrapcb_dataset, 
                               path + simstudy_dataset,
                               path + 'fig/',
                               model_class=model, 
                               theta_0=theta_0)

# results_test.data_per_run = np.array([{'endog': norm.rvs(size=50)} for _ in range(500)])
########################################################################
# results_test.data_per_run = results_test.data_per_run[:100]
# results_test.data_per_run[0]['endog'] = results_test.data_per_run[0]['endog'][:30]

# get bootstrap dataset
# nrep = 1000 # per expirience bs is accurate at 1000 reps
# parametric = True
# results_test.generate_bs_dataset(nrep=nrep, parametric=parametric)
results_test.init_dataset(show_first_run=False)
########################################################################
# results_test.fitresults_per_run = results_test.fitresults_per_run[:100]

### show
results_test.get_information_for_dataset()
# results_test.show_dataset_at_run(0, show_seperated=True, save_figures=False)
######################################################################## 
# results_test.results.set_results_from_dict(results_test.fitresults_per_run[0]) 
# results_test.data_plots(nx=100, xmin=-3, xmax=3)
# results_test.confregion_plots()
# plt.show()
# results_test.confband_plots(nx=50, xmin=-3, xmax=3)
# plt.show()

### simstudy
xs = np.linspace(-1.8, 1.8, 10)
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
    'method': 'delta',
    'test':   'w'
},{
    'method': 'delta',
    'test':   'studw'
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
                      which='both', show=False)