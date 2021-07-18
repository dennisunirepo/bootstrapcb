This Python package contains an extension to the Python library statsmodels that mainly implements the bootstrap approaches in order to construct confidence bands for a distribution function. The main component of this package is the function 'confband', which is available after fitting a parametric model to the data via a result object. The parametric model thereby needs to be implemented via a subclass of 'bootstrapcbModel' and by specifying a local distribution using one of the generic distribution classes 'rv_continous' or 'rv_discrete' from the scipy.stats package.

For a demonstration run the file 
```
python demo.py
```
