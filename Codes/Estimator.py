# Estimating the parameters: alpha_1, cv0, cv_bar0, eta, s_up, s_f, c_i, sigma, rate using dual_annealing method
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import integrate, optimize, interpolate
from scipy.ndimage.interpolation import shift
import time
import numpy.random as random
import multiprocessing
from multiprocessing import Pool
import sys
import os

#  ################################# Data and Results Path ###########################################
dirname = os.path.dirname(__file__)
mainPath = os.path.join(dirname,'USA/')
dataPath = 'Data/'
resultPath = 'Result/'
dataFile = mainPath + dataPath +'MainData.csv' # the csv file containing
# Location (the two-letter abbreviation)	Admin_Dose_1_Cumulative (accumulative administered dose 1)	PopElig
# (number of delivered vaccine doses to be administered as a first dose used as an approximation for eligible population)
# new_case (weekly new reported cases)	new_death (weekly new reported deaths associated with COVID-19) 	population
# (total population aged 5 and older) popf16 (population aged 16 and older as an approximate for adults population)

hesitancyFile = mainPath + dataPath + 'hesitancy_variaty.csv' # the csv file containing
#Location (the two-letter abbreviation)	Estimated_Hesitant_Unsure (estimated proportion of hesitant or unsure adults)
# Estimated_Hesitant (estimated proportion of hesitant  adults)	Estimated_Strongly_Hesitant (estimated proportion of
# strongly hesitant where we name them as vaccine refusers)


Fit_pow = mainPath + dataPath + 'fit_pow.csv' #the csv file containing cv_cof (the exponent of power law function \
# obtained from fitting a power law function to survey data regarding concerns about vaccine side effects) and
# Location (the two-letter abbreviation)
# reading the csv files
fitPow = pd.read_csv(Fit_pow)
hesitancy = pd.read_csv(hesitancyFile)
OWID = pd.read_csv(dataFile)
OWID = OWID.reset_index()
OWID['Date'] = pd.to_datetime(OWID['Date'])
method0 = 'LSODA' #method for solving ODEs one of the arguments of integrate.solve_ivp
StateName = list(OWID['Location'].unique()) #extracting the name of jurisdictions
iter_interval = 1000 #maximum number of iteration, dual-annealing optimization algorithm's control variables (max_iter)
initalTemp = 50000 #initial temperature, dual-annealing optimization algorithm's control variables (inital_temp)


def system_dynamics(x, y, alpha, cv1, cbar, c_i,  k1,  *arg_N):
    leng = 5
    lenn, N, fr_Ne, Ntt, cvcof =arg_N[0:leng]
    time_ref = arg_N[leng:leng + lenn]
    number_of_case = arg_N[leng + lenn: leng + 2 * lenn]
    number_of_death = arg_N[leng + 2 * lenn: leng + 3 * lenn]
    frac_avail_dose = arg_N[leng + 3 * lenn: leng + 4 * lenn]

    number_of_case_interpolated = interpolate.interp1d(time_ref, number_of_case, kind='zero', axis=0, fill_value=
    "extrapolate")
    number_of_death_interpolated = interpolate.interp1d(time_ref, number_of_death, kind='zero', axis=0, fill_value=
    "extrapolate")
    frac_avail_dose_interpolated = interpolate.interp1d(time_ref, frac_avail_dose, kind='zero', axis=0, fill_value=
    "extrapolate")
    im_vax, br_vax, = y
    totalvax = im_vax + br_vax  #proportion of vaccinated
    frac_avail_dose_interpolated = frac_avail_dose_interpolated(x) - totalvax #total available doses at time x /N
    if frac_avail_dose_interpolated < 0.0:
        frac_avail_dose_interpolated = 0.0
    cv = cv1*((x-time_ref[0])*0.5+1)**cvcof
    excess_payoff = -cv + cbar + c_i * (number_of_case_interpolated(x) / Ntt) +\
                    (number_of_death_interpolated(x) / Ntt)
    # Compute intermediate values
    intermediate_im = (1 - alpha) * fr_Ne - im_vax
    intermediate_br = alpha * fr_Ne - br_vax
    # Calculate num_im_reg_m, num_im_reg_w, num_br_reg_m, and num_br_reg_w using the conditions
    conditions = {
        'num_im_reg': intermediate_im * totalvax *  excess_payoff if excess_payoff > 0.0 and
                                                                                intermediate_im > 0 else 0.0,
        'num_br_reg': intermediate_br if excess_payoff > 0.0 and intermediate_br > 0 else 0.0
    }
    # Calculate total_reg
    total_reg = sum(conditions.values())

    # Calculate dot_num_im_vax_m, dot_num_br_vax_m, dot_num_im_vax_w, and dot_num_br_vax_w
    if total_reg > frac_avail_dose_interpolated:
        factor = frac_avail_dose_interpolated * k1 / total_reg
    else:
        factor = k1
    dot_num_im_vax = factor * conditions['num_im_reg']
    dot_num_br_vax = factor * conditions['num_br_reg']


    return dot_num_im_vax, dot_num_br_vax


def obj_ode(parameters, *arg): #
    number_of_elements = 5
    ssize, N, fr_Ne, Ntot, cvcof = arg[0:number_of_elements]
    arg_N = [ssize, N, fr_Ne, Ntot, cvcof]
    Time_Ref = arg[number_of_elements: number_of_elements + ssize]
    new_vaccinated = arg[number_of_elements + ssize:number_of_elements + 2 * ssize]
    Number_of_Case = arg[number_of_elements + 2 * ssize:number_of_elements + 3 * ssize]
    Number_of_Death = arg[number_of_elements + 3 * ssize:number_of_elements + 4 * ssize]
    Fraction_of_Eligible = arg[number_of_elements + 4 * ssize:number_of_elements + 5 * ssize]
    alpha_1_1, cv1,  cv_bar01, tilde_c_i_1,  k1  = parameters
    sol = integrate.solve_ivp(system_dynamics, [Time_Ref[0], Time_Ref[-1]], (
                            0, 0), args=(alpha_1_1, cv1, cv_bar01, tilde_c_i_1,
                                                                              k1,  *arg_N, *Time_Ref,
                                                                             *Number_of_Case, *Number_of_Death,
                                                                             *Fraction_of_Eligible),
                          t_eval=Time_Ref, dense_output=True, method=method0)
    model_output = N* sol.y[0, :] + N*sol.y[1, :]
    model_output_shifted = np.insert(model_output, 0, 0)
    model_output_shifted = np.delete(model_output_shifted, [-1])
    ee = np.asarray(model_output - model_output_shifted)
    residuals_22 = ee - np.asarray(new_vaccinated)
    RSE_22 = np.sum(residuals_22 ** 2)
    return RSE_22


def optm(arg):
    i, seed_initial = arg
    Ref_Time = []
    total_vaccinated = (OWID.loc[OWID['Location'] == i, 'Admin_Dose_1_Cumulative']).to_numpy(dtype=np.float64)
    total_shifted = np.insert(total_vaccinated, 0, 0)
    total_shifted = np.delete(total_shifted, [-1])
    new_vaccinated = total_vaccinated - total_shifted
    Number_of_Case = OWID.loc[OWID['Location'] == i, 'new_case'].to_numpy(dtype=np.float64)
    Number_of_Death = OWID.loc[OWID['Location'] == i, 'new_death'].to_numpy(dtype=np.float64)
    death_case_fraction = Number_of_Death/Number_of_Case
    min_death_case = np.min(death_case_fraction)
    if min_death_case > 0.0:
        max_ci_scaled = min_death_case
    else:
        non_min_val = death_case_fraction[death_case_fraction !=min_death_case]
        max_ci_scaled = np.min(non_min_val)
    total_doses = OWID.loc[OWID['Location'] == i, 'dose_distributed'].to_numpy(dtype=np.float64)
    N = OWID.loc[OWID['Location'] == i, 'popf12'].iloc[0]
    Nf = OWID.loc[OWID['Location'] == i, 'popf16'].iloc[0]
    Ntotal = OWID.loc[OWID['Location'] == i, 'poptotal'].iloc[0]
    cv_cof = fitPow.loc[fitPow['Location'] == i, 'cv_cof'].iloc[0]
    NonHesitant = N - Nf*hesitancy.loc[hesitancy['Location'] == i, 'Estimated_Strongly_Hesitant'].iloc[0]
    fraction_doses = np.clip(total_doses/N, 0, None)
    Ref_Time_2 = list(OWID.loc[OWID['Location'] == i, 'Date'])
    fr_Ne = NonHesitant/N #proportion of population who are not hesitant
    for k in Ref_Time_2:
        delta = k - Ref_Time_2[0]
        Ref_Time.append(delta.days)
    Time_Ref = np.array(Ref_Time, dtype=np.float64)
    Time_Ref = np.divide(Time_Ref, 7)
    max_alpha, min_alpha, max_cv, min_cv, max_cbar0, min_cbar0 = 1.0, 0.0, 1.0, 0.0, 1.0, 0.0  # defining the upper and lower
    max_ci, min_ci, max_cd, min_cd, max_rate, min_rate = max_ci_scaled, 0.0, 1.0, 0.0, 10.0, 0.0
    np.random.seed(seed_initial)
    cv_g = random.random() #alpha_1_1, cv1,  cv_bar01, tilde_c_i_1, cd_1, k1
    cv_bar0_g = max_cbar0*random.random()
    tilde_C_I_g = 0 + max_ci * random.random()
    rate_g = 0 + max_rate * random.random()
    alpha_g = 0 + random.random()
    seed0 = seed_initial
    int_guess = np.array([alpha_g, cv_g,  cv_bar0_g,  tilde_C_I_g,   rate_g])
    limit_list = [(min_alpha,max_alpha), (min_cv,max_cv), (min_cbar0, max_cbar0),
                  (min_ci, max_ci),  (min_rate,max_rate)]
    startTime = time.time()
    res_opt = optimize.dual_annealing(obj_ode, bounds=limit_list, x0=int_guess,  maxiter=
        iter_interval, seed=random.default_rng(seed=seed0), initial_temp=50000, args=[len(Time_Ref), N, fr_Ne, Ntotal,
                                                                                      cv_cof,  *Time_Ref, *new_vaccinated,
                                                              *Number_of_Case, *Number_of_Death,
                                                              *fraction_doses])

# Extracting the estimated parameters alpha_1_1, cv1,  cv_bar01, tilde_c_i_1, cd_1, k1

    alpha_1, cv,  cv_bar0, tilde_c_i,   rate= res_opt.x
    RSE_1 = res_opt.fun



    with open(mainPath + resultPath +'.txt', "a") as text_file:
            print(f"{i}, alpha_1: {alpha_1}, cv: {cv}, cv_bar0: {cv_bar0},  "
                  f"c_i: {tilde_c_i},  "
                  f"rate: {rate},"
                  f" RSE: {RSE_1}, seed:{seed_initial}, ElapsedTime: {time.time()-startTime}", file=text_file)
    arg_N = [len(Time_Ref), N, fr_Ne, Ntotal, cv_cof]
    sol = integrate.solve_ivp(system_dynamics, [Time_Ref[0], Time_Ref[-1]], (
        0, 0), args=(alpha_1, cv, cv_bar0, tilde_c_i, rate, *arg_N, *Time_Ref,
                     *Number_of_Case, *Number_of_Death, *fraction_doses), t_eval=Time_Ref,
                              dense_output=True, method=method0)
    model_output = N * (sol.y[0, :] + sol.y[1, :])
    model_output_shifted = np.insert(model_output, 0, 0)
    model_output_shifted = np.delete(model_output_shifted, [-1])
    new_vaccinated_fitted = np.asarray(model_output - model_output_shifted)
    simulatedFile = pd.DataFrame(np.reshape(np.asarray([i] * len(Time_Ref)), (len(Time_Ref), 1)),
                                 columns=['Location'])
    simulatedFile['new_vaccinated_fitted'] = np.reshape(new_vaccinated_fitted, (len(Time_Ref), 1))
    simulatedFile['new_vaccinated'] = np.reshape(new_vaccinated, (len(Time_Ref), 1))
    simulatedFile['time_ref'] = np.reshape(Time_Ref, (len(Time_Ref), 1))
    filepath = Path(mainPath + resultPath + i + '_SimFile.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    simulatedFile.to_csv(filepath)

    return i, alpha_1, cv,  cv_bar0, tilde_c_i,  rate,  RSE_1, seed_initial


def main():
    num_runs = 5 #number of different seeds

    result_dataframe = pd.DataFrame( columns=['Location', 'alpha_1', 'cv',  'cvbar0',  'c_i',
                                    'rate','RSE_1','seed'])
    initial_seed = 2023
    Number_of_cpus = multiprocessing.cpu_count()
    for run in range(num_runs):
        seed_run = initial_seed + run
        seed_list = [seed_run]*len(StateName)
        with Pool(processes=Number_of_cpus) as run_pool:
            parallel_output = run_pool.map(optm, zip(StateName, seed_list))
            run_pool.close()
            run_pool.join()
        ddf = pd.DataFrame(parallel_output, columns=['Location', 'alpha_1', 'cv',  'cvbar0',  'c_i',
                                    'rate', 'RSE_1', 'seed'])
        result_dataframe = pd.concat([ddf,result_dataframe])
    filepath = Path(mainPath + resultPath +'_estimated.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    result_dataframe.to_csv(filepath)


if __name__ == '__main__':
    main()



