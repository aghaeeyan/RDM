#For the sake of simplicity, this code disregard the seed. However, the reported results in the paper--including the parameter estimation
# and inference are based the best seed--in terms of resultant error--which is available on file mechanistic_model_0_1.csv
import pandas as pd
import numpy.random as random
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import integrate, optimize, interpolate
import statistics
from scipy.ndimage.interpolation import shift
from statsmodels import regression
from scipy.stats import pearsonr
import sys
import os
import copy
from scipy.stats import normaltest

dirname = os.path.dirname(__file__)
PathFileSave =  dirname + 'Result/BootData/'
NdataBoot = 500  # Number of bootstrap data
maxAR_order = 10  # AutoRegressive Order
inp =sys.argv # two letter abbreviation of the jurisdictions as input


i = str(inp[1])
fitFile = dirname + '/Result/' +  i +'_SimFile.csv'  # contains 'new_vaccinated_fitted' (Model output), 'new_vaccinated' (Data),
# 'time_ref' (Number of Week)
df = pd.pd.read_csv(fitFile)

nData = df['time_ref'].value_counts().loc[i] #number of datapoints for jurisdiction i
new_vaccinated = df.loc[df['Location'] == i, 'new_vaccinated'].to_numpy(dtype=np.float64)
new_vaccinated_fitted = df.loc[df['Location'] == i, 'new_vaccinated_fitted'].to_numpy(dtype=np.float64)
time_ref = df.loc[df['Location'] == i, 'time_ref'].to_numpy(dtype=np.float64)


################### NON-PARAMETRIC BOOTSTRAPPING APPROACH #########################
residuals =  new_vaccinated -  new_vaccinated_fitted  # calculating the residuals
mean_residual = statistics.mean(residuals)
residuals = residuals - mean_residual # obtaining zero mean residuals
# calculating the order of AR (less or equal maxAR_order) for which the correlation between the resultant
# innovation and its first order lag is minimum
ARvalue = [0.0] * maxAR_order
for ord in range(maxAR_order):
    residuals_shift1 = np.empty([ord + 1, len(residuals)])
    for p in range(ord + 1):
        residuals_shift1[p] = shift(residuals, p + 1, cval=0)
    [param1, resSD] = regression.linear_model.yule_walker(residuals, order=ord + 1)
    residuals_un_cor1 = copy.deepcopy(residuals)
    for p in range(ord + 1):
        residuals_un_cor1 = -param1[p] * residuals_shift1[p] + residuals_un_cor1
    residuals_un_cor1 = residuals_un_cor1[ord + 1:]
    mean_residuals_un_cor1 = statistics.mean(residuals_un_cor1)
    residuals_un_cor1 = residuals_un_cor1 - mean_residuals_un_cor1
    [aa, bb] = pearsonr(residuals_un_cor1, shift(residuals_un_cor1, 1, cval=0))
    ARvalue[ord] = aa
ARvalue = list(np.abs(ARvalue))
AR_order = ARvalue.index(min(ARvalue)) + 1 # the best AR order is obtained
# calculating the resultant residual which is equal to residual minus AR(AR_order)
residuals_shift = np.empty([AR_order + 1, len(residuals)])
for p in range(AR_order):
    residuals_shift[p] = shift(residuals, p + 1, cval=residuals[0])
[param, resSD] = regression.linear_model.yule_walker(residuals, order=AR_order)
residuals_un_cor = copy.deepcopy(residuals)
for p in range(AR_order):
    residuals_un_cor = -param[p] * residuals_shift[p] + residuals_un_cor
residuals_un_cor = residuals_un_cor[AR_order:] # the resultant residuals obtained
rng = np.random.default_rng()
# creating residuals_pool to randomly resample
residuals_pools = rng.choice(residuals_un_cor, size=(NdataBoot, nData))
bootstrapped_data = np.empty([NdataBoot, nData])
a = 0.0*np.ones(len(residuals),)
for p in range(AR_order):
    a = param[p] * residuals_shift[p] + a
for kk in range(NdataBoot):
    uu = a  + residuals_pools[kk]
    bootstrapped_data[kk] = (new_vaccinated_fitted + uu + mean_residual)
    bootstrapped_data[bootstrapped_data < 0] = 0 # To avoid having non-negative data in synthesized dataset
df3_pr = pd.DataFrame(np.transpose(bootstrapped_data), columns=list(map(str,
                                                                              range(0,
                                                                                         bootstrapped_data.shape[0]))))
df3_pr['time'] = time_ref
bootpath = PathFileSave + 'non/'  + i + '.csv'
filepath = Path(bootpath)
filepath.parent.mkdir(parents=True, exist_ok=True)
# df3_pr.to_csv(filepath)
################### PARAMETRIC BOOTSTRAPPING APPROACH #########################

rng = np.random.default_rng()
dist = np.empty([nData, NdataBoot])
new_vaccinated_fitted[new_vaccinated_fitted<0] = 0 # To make sure there is no invalid data
for jj in range(nData):
    dist[jj]= rng.poisson(new_vaccinated_fitted[jj],NdataBoot) # for each time instant, NdataBoot samples are drawn from Poisson
    # distribution whose mean equals the estimated number of new vaccinated individuals
df3_pr = pd.DataFrame(dist, columns=list(map(str, range(0, dist.shape[1]))))
df3_pr['time'] = time_ref
bootpath = PathFileSave + 'pois/' + i + '.csv'
filepath = Path(bootpath)
filepath.parent.mkdir(parents=True, exist_ok=True)
    # df3_pr.to_csv(filepath)
