# RDM
Supplementary data and code associated with the manuscript "Revealing the unseen: More than 30% of Americans relied on others' experience when deciding on taking the COVID-19 vaccine"

DOI 10.5281/zenodo.7649937

https://zenodo.org/badge/latestdoi/600241718
In this repository you may find the data and the codes necessary to replicate the reported results in our manuscript, "Revealing the unseen: More than 30% of Americans relied on others' experience when deciding on taking the COVID-19 vaccine".
This readme walks you through the steps need to be taken to run the codes.
The Python code used here also depends on several packages, which are listed below:

Pandas
Numpy
Datetime
Pathlib
Scipy (integrate, optimize, interpolate, ndimage.interpolation, stats)
Time
re
statsmodels
Multiprocessing*
Sys*
Os*

(*) are not necessary for getting results. However, if you are going to download and run the codes without modifications, you need those packages installed on your PC.

You need to have Python and subsequently those mentioned packages installed on your PC.

The folder Data should be put in a folder named “USA” in the same directory as the folder Codes are. 
In addition, an empty folder named “Result” should be created in the folder “USA”.

##Codes:

Estimator.py: This script estimates and returns the estimated parameters for the 51 jurisdictions of the US. To run the code, you just need to type “python Estimator.py”. 

BootstrappedDataGenerator.py: This script creates the synthetic datasets to be used in bootstrap methods. The output of the code is two synthetized datasets one based on non-parametric residual resampling with replacement and the other one is based on parametric bootstrap approach. To do so, you need type in the command line “python BootstrappedDataGenerator.py i”, where “i” is the two-letter abbreviation of the jurisdiction. Number of bootstrapped data is set 450. However, the reported results are for 500.

Bootstrap.py: After obtaining the bootstrapped dataset, you may run this script to obtain the estimated parameters for that dataset.


CIcalculator.py: This script calculated the 95% CI based on percentile approach.


## Data
MainData: Contains the polished temporal data originally obtained from Centers for Disease Control and Prevention (CDC) and US Census Bureau as cited in the manuscript.
Fit_pow: Contains the estimated value for cv_cof as detailed in the manuscript.
Hesitancy_variaty: contains the estimated hesitant population obtained from “Predicted Vaccine Hesitancy by State, PUMA, and County” (https://aspe.hhs.gov/reports/vaccine-hesitancy-covid-19-state-county-local-estimates) where also cited in the manuscript.

Mandate_date: Contains the starting date from which differentiating policies based on vaccination status were in effect. The dates were obtained from https://doi.org/10.1038/s41562-021-01079-8 where also cited in the manuscript.
