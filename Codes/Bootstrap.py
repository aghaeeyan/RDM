
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import integrate, optimize, interpolate
import time
import numpy.random as random
import multiprocessing
from multiprocessing import Pool
import sys
import os
from scipy.ndimage.interpolation import shift

#  ################################# Data and Results Path ###########################################
dirname = os.path.dirname(__file__)
mainPath = os.path.join(dirname,'USA/')
dataPath = 'Data/'
resultPath = 'Result/BootResult/'
inp = sys.argv
dataFile = mainPath + dataPath +'MainData.csv' # the csv file containing
# Location (the two-letter abbreviation)	Admin_Dose_1_Cumulative (accumulative administered dose 1)	PopElig
# (number of delivered vaccine doses to be administered as a first dose used as an approximation for eligible population)
# new_case (weekly new reported cases)	new_death (weekly new reported deaths associated with COVID-19) 	population
# (total population aged 5 and older) popf16 (population aged 16 and older as an approximate for adults population)

hesitancyFile = mainPath + dataPath + 'hesitancy_variaty.csv' # the csv file containing
#Location (the two-letter abbreviation)	Estimated_Hesitant_Unsure (estimated proportion of hesitant or unsure adults)
# Estimated_Hesitant (estimated proportion of hesitant  adults)	Estimated_Strongly_Hesitant (estimated proportion of
# strongly hesitant where we name them as vaccine refusers)

MandateFile = mainPath + dataPath + 'mandate_date.csv' # the csv file containing
# Location (the two-letter abbreviation) set_date (the date from which the new differentiating policies were in effect)
# announcement (20 days prior to implementing the policies)
Fit_pow = mainPath + dataPath + 'fit_pow.csv' #the csv file containing cv_cof (the exponent of power law function \
# obtained from fitting a power law function to survey data regarding concerns about vaccine side effects) and
# Location (the two-letter abbreviation)
# reading the csv files
fitPow = pd.read_csv(Fit_pow)
OWID = pd.read_csv(dataFile)
hesitancy = pd.read_csv(hesitancyFile)
mandate = pd.read_csv(MandateFile)
OWID = OWID.reset_index()
OWID['Date'] = pd.to_datetime(OWID['Date'])
method0 = 'RK45'
max_eta, min_eta, max_s, min_s, min_s_f, max_s_f, max_cbar0, max_C_I, max_sigma, max_rate =10.0, 1.01, 1.0, 0.0, \
0.0,1.0, 0.1, 1, 1, 10 # setting the upper and lower bounds of free parameters
inp = sys.argv
i = str(inp[1]) # two-letter abbreviation of the jurisdiction

dataFile2 = mainPath + 'Result/BootData/'+ 'non/'  + i + '.csv'
# dataFile2 = mainPath + 'Result/BootData/'+ 'pois/'  + i + '.csv' if parametric bootstrapping
bootstrap_pool = pd.read_csv(dataFile2)
number_of_case = OWID.loc[OWID['Location'] == i, 'new_case'].to_numpy(dtype=np.float64)
number_of_death = OWID.loc[OWID['Location'] == i, 'new_death'].to_numpy(dtype=np.float64)
N = OWID.loc[OWID['Location'] == i, 'population'].iloc[0]
dose1 = OWID.loc[OWID['Location'] == i, 'PopElig'].to_numpy(dtype=np.float64)
dose = shift(dose1, -1, mode='nearest')
Nf = OWID.loc[OWID['Location'] == i, 'popf16'].iloc[0]
mandate['announcement'] = pd.to_datetime(mandate['announcement'])
Nboot = 500  # Number of synthesized data
iter_interval = 1000
cv_cof = fitPow.loc[fitPow['Location'] == i, 'cv_cof'].iloc[0]
NonHesitant = N - Nf * hesitancy.loc[hesitancy['Location'] == i, 'Estimated_Strongly_Hesitant'].iloc[0]
Nn = NonHesitant/N
fraction_of_dose = np.clip(dose / N, 0, None)
ref_time2 = list(OWID.loc[OWID['Location'] == i, 'Date'])
announcement_date = mandate.loc[mandate['Location'] == 'US_'+ i, 'set_date'].iloc[0]
indicator, indicator1, ref_time = [],[],[]
for k in ref_time2:
    ManDate = pd.to_datetime(announcement_date)

    if k > ManDate:
        indc = 1
    else:
        indc = 0
    indicator.append(indc)
    delta = k - ref_time[0]
    ref_time.append(delta.days)
time_ref = np.divide(np.array(ref_time, dtype=np.float64),7)
inArg = np.argmax(np.asarray(indicator))
inArg1 = np.argmax(np.asarray(indicator1))

number_of_case_interpolated = interpolate.interp1d(time_ref, number_of_case, kind='zero', axis=0, fill_value=
"extrapolate")
number_of_death_interpolated = interpolate.interp1d(time_ref, number_of_death, kind='zero', axis=0, fill_value=
"extrapolate")
frac_avail_dose_interpolated1 = interpolate.interp1d(time_ref, fraction_of_dose, kind='zero', axis=0,
                                                         fill_value=
                                                         "extrapolate")



def dot_fraction_of_vaccinated(x, y, alpha_11, cv1, cv_bar0, eta, s_up, s_f, C_i, sigma_11, rate,):
    im_vax, br_vax = y
    cv_bar_t = cv_bar0 + (x - time_ref[inArg]) * s_up
    t_r = time_ref[inArg] + (eta - 1) * cv_bar0 / s_up
    cv_bar_f = eta * cv_bar0 - s_f * (x - t_r)
    if x < time_ref[inArg]:
        cbar = cv_bar0
    elif x < t_r:
        cbar = cv_bar_t
    else:
        cbar = max(cv_bar_f, 0)

    cv = cv1 * ((x - time_ref[0]) * 0.5 + 1) ** cv_cof
    frac_avail_dose_interpolated = frac_avail_dose_interpolated1(x) - im_vax - br_vax
    if frac_avail_dose_interpolated < 0.0:
        frac_avail_dose_interpolated = 0.0

    excess_payoff = -cv + cbar + C_i * (number_of_case_interpolated(x) / N) + \
                    (number_of_death_interpolated(x) / N)

    if excess_payoff > 0 and (1 - alpha_11) * Nn - im_vax > 0:
        num_im_reg = ((1 - alpha_11) * Nn - im_vax) * \
                     (im_vax + br_vax) * sigma_11 * excess_payoff
    else:
        num_im_reg = 0.0

    if excess_payoff > 0 and (alpha_11 * Nn - br_vax) > 0:
        num_br_reg = (alpha_11 * Nn - br_vax)
    else:
        num_br_reg = 0.0

    total_reg = num_im_reg + num_br_reg
    if total_reg > frac_avail_dose_interpolated:
        dot_num_im_vax = frac_avail_dose_interpolated * rate * num_im_reg / total_reg
        dot_num_br_vax = frac_avail_dose_interpolated * rate * num_br_reg / total_reg
    else:
        dot_num_im_vax = rate * num_im_reg
        dot_num_br_vax = rate * num_br_reg

    return dot_num_im_vax, dot_num_br_vax



def OBJ_ODE(parameters, *arg):
    ssize = arg[0]
    Time_Ref = arg[1:1 + ssize]
    new_vaccinated = arg[1 + ssize:1 + 2 * ssize]
    alpha_1_1, cv1,  cv_bar01, eta1, s_up1, s_f1,  C_i_1, sigma_1, rate_1  = parameters
    sol = integrate.solve_ivp(dot_fraction_of_vaccinated, [Time_Ref[0], Time_Ref[-1]], (0.0, 0.0),
                              args=(alpha_1_1, cv1,  cv_bar01, eta1, s_up1, s_f1, C_i_1,
                                                                             sigma_1, rate_1 ),
                              t_eval=Time_Ref,
                              dense_output=True, method=method0)
    model_output = N*sol.y[0, :] + N*sol.y[1, :]
    model_output_shifted = np.insert(model_output, 0, 0)
    model_output_shifted = np.delete(model_output_shifted, [-1])
    ee = np.asarray(model_output - model_output_shifted)
    residuals_22 =  np.asarray(new_vaccinated) - ee
    RSE_22 = np.sum(residuals_22 ** 2)
    return RSE_22


def optm(arg):
    itr = arg
    bootstrapped_data = bootstrap_pool[str(itr)].to_numpy(dtype=np.float64)
    bootstrapped_data[bootstrapped_data <0] = 0
    np.random.seed(765)
    cv_g = random.random()
    cv_bar0_g = max_cbar0 * random.random()
    eta_g = min_eta + (max_eta - min_eta) * random.random()
    s_up_g = min_s + (max_s - min_s) * random.random()
    s_f_g = min_s_f + (max_s_f - min_s_f) * random.random()
    tilde_C_I_g = 0 + max_C_I * random.random()
    sigma_g = 0 + max_sigma * random.random()
    rate_g = 0 + max_rate * random.random()
    alpha_g = 0 + random.random()
    seed0 = 27
    int_guess = np.array([alpha_g, cv_g, cv_bar0_g, eta_g, s_up_g, s_f_g, tilde_C_I_g, sigma_g, rate_g])
    limit_list = [(0, 1), (0, 1), (0, max_cbar0), (min_eta, max_eta), (min_s, max_s), (min_s_f, max_s_f),
                  (0, max_C_I), (0, max_sigma), (0, max_rate)]

    res_opt = optimize.dual_annealing(OBJ_ODE, bounds=limit_list, x0=int_guess, maxiter=
            iter_interval, initial_temp=50000, seed=random.default_rng(seed=seed0), args=[len(time_ref), *time_ref,
                                                                                          *bootstrapped_data])

    # Extracting the estimated parameters
    alpha_1, cv, cv_bar0, eta, s_up, s_f,  C_i,  sigma, rate= res_opt.x
    alpha = alpha_1*NonHesitant/N

    with open(mainPath + resultPath + 'non_' + i + '.txt', "a") as text_file:
        print(f"Code: {i}, alpha: {alpha}, alpha_1: {alpha_1}, cv: {cv}, cv_bar0: {cv_bar0}, eta: {eta}, s_up: {s_up},"
              f" s_f: {s_f}, "
                  f"C_i: {C_i},  "
                  f"sigma: {sigma},rate: {rate},"
                  , file=text_file)
    return


def main():

    pool_list = list()
    Number_of_cpus = multiprocessing.cpu_count()

    for p2 in range(Nboot):
        pool_list.append(p2)

    with Pool(processes=Number_of_cpus) as run_pool:
        run_pool.close()
        run_pool.join()



if __name__ == '__main__':
    main()



