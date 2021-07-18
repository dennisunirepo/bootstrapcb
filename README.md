# bootstrapcb
An extension of the statsmodels likelihoodmodel for constructing confidence bands and a simulation study in the context of OMNeT++

## Setup
This project was built and tested with Ubuntu LTS version 20.04. Although the simulation example in the folder 'omnetpp/example_mg1' was created for OMNeT++ version 5.2.6, it can also be run in the more recent version 6. This is provided via the official Git repository of the OMNeT++ simulation library. 

## Contents
We provide the bootstrapcb package that implements bootstrapping methods to construct confidence bands for distribution functions.This is based on the work of <cite>[Cheng and Iles (1983)][1]</cite> and later work, e.g. <cite>[Cheng (2005)][2]</cite>, which presents methods that use bootstrap to construct exact confidence regions in parameter space and then exploit these regions to a confidence band for the distribution. Construction of confidence bands for any other parametric model is also possible. For example, run 
```
python demo.py
```
to get an idea of how to use the package.

## OMNeT++ simstudy
A work-in-progress simulation study using the bootstrapcb package in the context of OMNeT++ can be found at 'omnetpp/example_mg1'. To create and run the complete simulation including a simple analysis of the results in Pyhton, execute the 'run' file from this folder. In addition, we provide a scave template under the 'omnetpp/scave_templates' folder that embeds the bootstrapcb package into the IDE of the current version. To do this, the files from this directory must be placed in the appropriate directories of the IDE. An example of a prebuild result file is 'MM1.anf'. Copy this to the 'simulations' directory or any directory that contains a 'result' folder with simulation output. Result analysis in OMNeT++ version 5.2.6 is done either directly by using the somewhat simpler tools via the IDE. Or more conveniently, by exporting the result to a CSV file and using the pandas library for data exploration. Python scripts demonstrating this can also be found at 'omnetpp/example_mg1/simulations'. For example, run 
```
python analysis.py
``` 
from there to get an idea. It seems that the command line tool scavotool is missing in the new version and there may also be bugs, so we recommend exporting the results from the simulation runs under version 5.2.6. Analysis with the tools integrated in the IDE can then still be done by simply copying the result files into a project under the newer version.

[1]: https://doi.org/10.2307/1267729
[2]: https://doi.org/10.1109/WSC.2005.1574257
