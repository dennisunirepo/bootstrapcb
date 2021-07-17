import math
import numpy as np
from scipy.stats import norm, expon, gamma, rv_continuous
from omnetpp.scave import results, chart, utils, plot, vectorops as ops
from bootstrapcb import bootstrapcbModel

# get chart properties
props = chart.get_properties()
utils.preconfigure_plot(props)

# collect parameters for query
filter_expression = props["filter"]
start_time = float(props["vector_start_time"] or -math.inf)
end_time = float(props["vector_end_time"] or math.inf)

# collect data
try:
    df = results.get_vectors(filter_expression, include_attrs=True, include_itervars=True)
except ValueError as e:
    plot.set_warning("Error while querying results: " + str(e))
    exit(1)

if df.empty:
    plot.set_warning("The result filter returned no data.")
    exit(1)

##################################################
# parse functions in probs to look like this:
#
# class model(bootstrapcbModel):
#   # local distribution class
#   def __init__(self, endog, exog=None, start_params=None)
#     self.dist = norm
#     super(model, self).__init__(endog=endog, exog=exog, start_params=start_params)
#
#   or
#
#   # local distribution class
#   class distribution(rv_continuous):     
#     def _cdf(self, x, mu, sigma):
#       a = mu**2 / sigma**2
#       b = sigma**2 / mu
#       return gamma.cdf(x, a, scale=b) 
#     def _pdf(self, x, mu, sigma):
#       a = mu**2 / sigma**2
#       b = sigma**2 / mu
#       return gamma.pdf(x, a, scale=b) 
#     def _argcheck(self, mu, sigma):
#       return sigma >= 0
# 
#   # function for which the confidence band is to be calculated
#   def display_func(self, x, *args, **kwds):
#     return self.cdf(x, *args, **kwds)
#
params_str = props["params"]
distribution_type_str = props["distribution_type"]
cdf_str = props["cdf"] 
pdf_str = props["pdf"]
argcheck_str = props["argcheck"]
test_str = props["test"]
method_str = props["method"]
f_str = props["display_func"]
if not cdf_str and not pdf_str: 
    plot.set_warning("Distribution not defined")
    exit(1)
# parse theta0
if params_str:
    parsed_params, theta0_str = params_str.split(" = ")
    theta0 = eval('['+theta0_str+']')
parsed_code             =   "class model(bootstrapcbModel):\n"  

if distribution_type_str == "custom":
    parsed_code        +=   "  class distribution(rv_continuous):\n" 
    # parse cdf 
    if cdf_str: 
        code = cdf_str.split("\n")
        parsed_code     +=  "    def _cdf(self, x, {}):\n".format(parsed_params)
        for line in code: 
            parsed_code +=  "      "+line+"\n"
    # parse pdf 
    if pdf_str: 
        code = pdf_str.split("\n")
        parsed_code     +=  "    def _pdf(self, x, {}):\n".format(parsed_params) 
        for line in code: 
            parsed_code +=  "      "+line+"\n"
    # parse argcheck
    if argcheck_str: 
        code = argcheck_str.split("\n")
        parsed_code     +=  "    def _argcheck(self, {}):\n".format(parsed_params) 
        for line in code: 
            parsed_code +=  "      "+line+"\n"
else:
    parsed_code         +=( "  def __init__(self, endog, exog=None, start_params=None):\n"
                        +   "    self.dist = {}\n".format(distribution_type_str)
                        +   "    super(model, self).__init__(endog=endog, exog=exog, start_params=start_params)\n" )

parsed_code         +=  "\n"
# parse display_func
if f_str:
    code = f_str.split("\n")
    parsed_code         +=  "  def display_func(self, x, *args, **kwds):\n"
    for line in code: 
        parsed_code     +=  "    "+line+"\n"
exec(parsed_code) 
# parse test
if test_str == "Chi2 quantiles":
    test = "approx"
elif test_str == "Wald-statistic":
    test = "w"
elif test_str == "Studentized wald-statistic":
    test = "studw"
elif test_str == "Likelihood-ratio-statistic":
    test = "lr"
# parse method
if method_str == "Delta method":
    method = "delta"
elif method_str == "Nelder-mead optimization":
    method = "nm"
elif method_str == "Uniform points on confindence region":
    method = "mc"
    
print("params: " + parsed_params)
print("start_params: {}".format(theta0))
print(parsed_code)
print(test)
print(method)
##################################################

# plot confband (only plotting one vector supported)
if len(df.index) > 1:
    plot.set_warning("Filter has too many values.")
    exit(1)
    
values = df.iloc[0]["vecvalue"]
tmpModle = model(values, start_params=theta0)
tmpResults = tmpModle.fit()
tmpMle = tmpResults.params
print(tmpResults.summary())

xs = np.linspace(tmpResults.model.ppf(0.01, *tmpMle), 
                 tmpResults.model.ppf(0.99, *tmpMle), 50)
ecdf        = tmpResults.model.ecdf(xs)
plot.plot(xs, ecdf, label='empirical distribution', 
                            drawstyle='steps-post')
fitted_func = tmpResults.mle(xs)
plot.plot(xs, fitted_func,    label='fitted distribution')
confband    = tmpResults.confband(xs, alpha=0.1, nrep=100, parametric=True,
                                  test=test, method=method)
plot.plot(xs, confband[:, 0], label='confidence band',
                              color='green', linestyle='--')
plot.plot(xs, confband[:, 1], color='green', linestyle='--')

utils.postconfigure_plot(props)

utils.export_image_if_needed(props)
utils.export_data_if_needed(df, props)