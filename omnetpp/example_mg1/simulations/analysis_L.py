from scipy.stats import geom, rv_discrete
from bootstrapcb import bootstrapcbModel, genericAnalysis

class model(bootstrapcbModel):
    # local distribution class
    class distribution(rv_discrete):     
        def _cdf(self, n, p):
            return geom.cdf(n+1, 1-p) 
        def _logcdf(self, n, p):
            return geom.logcdf(n+1, 1-p) 
        def _pmf(self, n, p):
            return geom.pmf(n+1, 1-p) 
        def _logpmf(self, n, p):
            return geom.logpmf(n+1, 1-p) 
        def _ppf(self, q, p):
            return geom.ppf(q, 1-p) - 1
        def _rvs(self, p, size=None, random_state=None):
            return geom.rvs(1-p, size=size, random_state=random_state)
        def _argcheck(self, p):
            return 0 < p and p < 1 

    # function for which the confidence band is to be calculated
    def display_func(self, x, *args, **kwds):
        return self.cdf(x, *args, **kwds)

# init analysis
ia, s = 1, 0.8
iar, sr = 1/ia, 1/s
theta_0 = [iar/sr] 
path = "datasets/"
sim_runs_dataset = "analysis_L.csv"
bootstrapcb_dataset = "analysis_L.json"
simstudy_dataset = "analysis_L.txt"
results_test = genericAnalysis(path + sim_runs_dataset,
                               path + bootstrapcb_dataset, 
                               path + simstudy_dataset,
                               path + 'fig_analysis_L/',
                               model_class=model, 
                               theta_0=theta_0)

# get run dataset
# statistic = "nCustomers"
# strategy = "equi_timedist"
# nbatches = 100
# results_test.load_data_from_sim_dataset(statistic=statistic, strategy=strategy, nbatches=nbatches)
########################################################################
# results_test.data_per_run = results_test.data_per_run[:1]
# results_test.data_per_run[0]['endog'] = results_test.data_per_run[0]['endog'][:100]

# get bootstrap dataset
# nrep = 1000 # per expirience bs is accurate at 1000 reps
# parametric = True
# results_test.generate_bs_dataset(nrep=nrep, parametric=parametric)
results_test.init_dataset(show_first_run=False)
########################################################################
# results_test.fitresults_per_run = results_test.fitresults_per_run[:100]

### show
results_test.get_information_for_dataset()
results_test.show_dataset_at_run(0, show_seperated=True, save_figures=False) 
######################################################################## 
# results_test.results.set_results_from_dict(results_test.fitresults_per_run[0]) 
# results_test.data_plots(nx=100, xmin=0, xmax=3)
# results_test.confregion_plots()
# plt.show()
# results_test.confband_plots(nx=50)
# plt.show()