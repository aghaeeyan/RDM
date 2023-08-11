
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


Fit_pow = mainPath + dataPath + 'fit_pow.csv' #the csv file containing cv_cof (the exponent of power law function \
# obtained from fitting a power law function to survey data regarding concerns about vaccine side effects) and
# Location (the two-letter abbreviation)
# reading the csv files
seed_file = mainPath + dataPath + 'mechanistic_model_0_1.csv'
fitPow = pd.read_csv(Fit_pow)
seed_read = pd.read_csv(seed_file)
hesitancy = pd.read_csv(hesitancyFile)
OWID1 = pd.read_csv(dataFile)
OWID1 = OWID1.reset_index()
OWID1['Date'] = pd.to_datetime(OWID1['Date'])
OWID = OWID1.sort_values('Date')
method0 = 'LSODA'
Nboot = 500  # Number of synthesized data
iter_interval = 1000
inp = sys.argv
i = str(inp[1]) # two-letter abbreviation of the jurisdiction

dataFile2 =  mainPath + 'Result/BootData/'+ 'non/'  + i + '.csv' # The file contained the bootstrapped data
bootstrap_pool = pd.read_csv(dataFile2)
number_of_case = OWID.loc[OWID['Location'] == i, 'new_case'].to_numpy(dtype=np.float64)
number_of_death = OWID.loc[OWID['Location'] == i, 'new_death'].to_numpy(dtype=np.float64)
frac_avail_dose = OWID.loc[OWID['Location'] == i, 'dose_distributed'].to_numpy(dtype=np.float64)
N = OWID.loc[OWID['Location'] == i, 'popf12'].iloc[0]
Nf = OWID.loc[OWID['Location'] == i, 'popf16'].iloc[0]
Ntotal = OWID.loc[OWID['Location'] == i, 'poptotal'].iloc[0]
cv_cof = fitPow.loc[fitPow['Location'] == i, 'cv_cof'].iloc[0]
NonHesitant = N - Nf*hesitancy.loc[hesitancy['Location'] == i, 'Estimated_Strongly_Hesitant'].iloc[0]
fr_Ne = NonHesitant / N
fraction_doses = np.clip(frac_avail_dose / N, 0, None)
Ref_Time_2 = list(OWID.loc[OWID['Location'] == i, 'Date'])
Ref_Time = []
death_case_fraction = number_of_death / number_of_case
min_death_case = np.min(death_case_fraction)
if min_death_case > 0.0:
    max_ci_scaled = min_death_case
else:
    non_min_val = death_case_fraction[death_case_fraction != min_death_case]
    max_ci_scaled = np.min(non_min_val)
max_alpha, min_alpha, max_cv, min_cv, max_cbar0, min_cbar0 = 1.0, 0.0, 1.0, 0.0, 1.0, 0.0  # defining the upper and lower
max_ci, min_ci, max_cd, min_cd, max_rate, min_rate = max_ci_scaled, 0.0, 1.0, 0.0, 10.0, 0.0
for k in Ref_Time_2:
    delta = k - Ref_Time_2[0]
    Ref_Time.append(delta.days)
Time_Ref = np.array(Ref_Time, dtype=np.float64)
Time_Ref = np.divide(Time_Ref, 7)
frac_avail_dose_interp = interpolate.interp1d(Time_Ref, fraction_doses, kind='zero', axis=0, fill_value=
"extrapolate")
number_of_case_interpolated = interpolate.interp1d(Time_Ref, number_of_case, kind='zero', axis=0, fill_value=
    "extrapolate")
number_of_death_interpolated = interpolate.interp1d(Time_Ref, number_of_death, kind='zero', axis=0, fill_value=
    "extrapolate")



def system_dynamics(x, y, alpha, cv1, cbar, c_i, k1):
    #alpha_g, cv_g, cv_bar0_g, tilde_C_I_g, cd_g, rate_g

    im_vax, br_vax, = y
    totalvax = im_vax + br_vax  #proportion of vaccinated
    frac_avail_dose_interpolated = frac_avail_dose_interp(x) - totalvax  # total available doses at time x /N
    if frac_avail_dose_interpolated < 0.0:
        frac_avail_dose_interpolated = 0.0
    cv = cv1*((x-Time_Ref[0])*0.5+1)**cv_cof
    excess_payoff = -cv + cbar + c_i * (number_of_case_interpolated(x) / Ntotal) +\
                    (number_of_death_interpolated(x) / Ntotal)
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


def OBJ_ODE(parameters, *arg):
    #alpha_g, cv_g, cv_bar0_g, tilde_C_I_g, cd_g, rate_g
    ssize = arg[0]
    new_vaccinated = arg[1:]
    alpha_1_1, cv1, cv_bar01,  tilde_c_i_1,  rate_1  = parameters
    sol = integrate.solve_ivp(system_dynamics, [Time_Ref[0], Time_Ref[-1]], (0.0, 0.0),
                              args=(alpha_1_1, cv1, cv_bar01, tilde_c_i_1,
                                                                             rate_1 ),
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
    itr = arg + Nboot
    bootstrapped_data = bootstrap_pool[str(itr)].to_numpy(dtype=np.float64)
    bootstrapped_data[bootstrapped_data <0] = 0
    np.random.seed(seed_read.loc[seed_read['Location'] == i, 'seed'].iloc[0])
    cv_g = random.random()
    cv_bar0_g = max_cbar0 * random.random()
    tilde_C_I_g = 0 + max_ci * random.random()
    rate_g = 0 + max_rate * random.random()
    alpha_g = 0 + random.random()
    seed0 = seed_read.loc[seed_read['Location'] == i, 'seed'].iloc[0]
    int_guess = np.array([alpha_g, cv_g, cv_bar0_g, tilde_C_I_g,  rate_g])
    limit_list = [(min_alpha, max_alpha), (min_cv, max_cv), (min_cbar0, max_cbar0),
                  (min_ci, max_ci),  (min_rate, max_rate)]
    try:

        res_opt = optimize.dual_annealing(OBJ_ODE, bounds=limit_list, x0=int_guess, maxiter=
            iter_interval, initial_temp=50000, seed=random.default_rng(seed=seed0), args=[len(Time_Ref),
                                                                                          *bootstrapped_data])
    # Extracting the estimated parameters
        alpha_1, cv,  cv_bar0, tilde_c_i,   rate= res_opt.x
        RSE = res_opt.fun
        error_flag = 0
        cd =1.0
    except:
        alpha_1, cv,  cv_bar0, tilde_c_i,  rate, RSE = [-1]*6
        error_flag = 1
        cd = 1.0

    with open(mainPath + dataPath + 'results/'+ str(iter_interval) + '.txt', "a") as text_file:
        print(f"Code: {i}, bootItem: {itr}, alpha_1: {alpha_1}, cv: {cv}, cv_bar0: {cv_bar0}, "
                  f"tilde_c_i: {tilde_c_i},  "
                  f"cd: {1.0},rate: {rate},"
                  f" RSE: {RSE},  ErrorFlag = {error_flag}", file=text_file)
    return i, itr, alpha_1, cv,  cv_bar0, tilde_c_i,  cd, rate, RSE

def main():

    pool_list = list()
    Number_of_cpus = multiprocessing.cpu_count()
    for p2 in range(Nboot):
        pool_list.append(p2)

    with Pool(processes=Number_of_cpus) as run_pool:
        parallel_output = run_pool.map(optm, pool_list)  # use tqdm to show the progress
        run_pool.close()
        run_pool.join()
    f = parallel_output
    ddf = pd.DataFrame(f, columns=['code', 'itr', 'alpha_1', 'cv', 'cv_bar0',
                                   'tilde_c_i',
                                   'cd', 'rate', 'RSE_1'])

    filepath = Path(mainPath + dataPath + 'results/'+ i +'.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ddf.to_csv(filepath)


if __name__ == '__main__':
    main()






