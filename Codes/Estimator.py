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
hesitancy = pd.read_csv(hesitancyFile)
mandate = pd.read_csv(MandateFile)
OWID = pd.read_csv(dataFile)
OWID = OWID.reset_index()
mandate['announcement'] = pd.to_datetime(mandate['announcement'])
OWID['Date'] = pd.to_datetime(OWID['Date'])
method0 = 'LSODA' #method for solving ODEs one of the arguments of integrate.solve_ivp
max_eta, min_eta, max_s, min_s, min_s_f, max_s_f, max_cbar0, max_C_I, max_sigma, max_rate =10.0, 1.01, 1.0, 0.0, \
0.0,1.0, 0.1, 1, 1, 10 # setting the upper and lower bounds of free parameters
StateName = list(OWID['Location'].unique()) #extracting the name of jurisdictions
iter_interval = 1000 #maximum number of iteration, dual-annealing optimization algorithm's control variables (max_iter)
initalTemp = 5000 #initial temperature, dual-annealing optimization algorithm's control variables (inital_temp)


def dot_fraction_of_vaccinated(x, y, alpha_11, cv1, cv_bar0, eta, s_up, s_f, C_i, sigma_11, rate_11, *arg_N):
    N, lenn, inArg, inArg1, cvcof = arg_N[0:5]
    time_ref = arg_N[5:5 + lenn]
    number_of_case = arg_N[5 + lenn: 5 + 2 * lenn]
    number_of_death = arg_N[5 + 2 * lenn: 5 + 3 * lenn]
    fraction_of_eligible = arg_N[5 + 3 * lenn:]
    number_of_case_interpolated = interpolate.interp1d(time_ref, number_of_case, kind='zero', axis=0, fill_value=
    "extrapolate")
    number_of_death_interpolated = interpolate.interp1d(time_ref, number_of_death, kind='zero', axis=0, fill_value=
    "extrapolate")
    fraction_of_eligible_interpolated = interpolate.interp1d(time_ref, fraction_of_eligible, kind='zero', axis=0, fill_value=
    "extrapolate")

    y0, y1 = y
    cv_bar_t = cv_bar0 + (x - time_ref[inArg])*s_up
    t_r = time_ref[inArg] + (eta - 1)*cv_bar0/s_up
    cv_bar_f = eta*cv_bar0 - s_f*(x-t_r)
    if x < time_ref[inArg]:
        cbar = cv_bar0
    elif x < t_r:
        cbar = cv_bar_t
    else:
        cbar = max(cv_bar_f, 0)
    if x < time_ref[inArg1]:
        cv = cv1
    else:
        cv = cv1*((x-time_ref[inArg1])*0.5+1)**cvcof


    payoff_excess_vaccinated2UnVaccinated = -cv + cbar + C_i * (number_of_case_interpolated(x) / N) + \
                                            (number_of_death_interpolated(x) / N)
    if payoff_excess_vaccinated2UnVaccinated > 0 and (1 - alpha_11) * fraction_of_eligible_interpolated(x) - y0 >0:
        dot_fraction_of_imitator_vaccinated = ((1 - alpha_11) * fraction_of_eligible_interpolated(x)  - y0) * \
                                            (y0 + y1) * (
                                                    sigma_11 * rate_11 * payoff_excess_vaccinated2UnVaccinated)
    else:
        dot_fraction_of_imitator_vaccinated = 0
    if payoff_excess_vaccinated2UnVaccinated > 0 and (alpha_11 * fraction_of_eligible_interpolated(x) - y1) > 0:
        dot_fraction_of_rationalist_vaccinated = rate_11 * (alpha_11  * fraction_of_eligible_interpolated(x) - y1)
    else:
        dot_fraction_of_rationalist_vaccinated = 0

    return dot_fraction_of_imitator_vaccinated, dot_fraction_of_rationalist_vaccinated


def obj_ode(parameters, *arg):
    ssize, N, inArg, inArg1, cvcof = arg[0:5]
    arg_N = [N, ssize, inArg, inArg1, cvcof]
    time_ref = arg[5:5 + ssize]
    new_vaccinated = arg[5 + ssize:5 + 2 * ssize]
    number_of_case = arg[5 + 2 * ssize:5 + 3 * ssize]
    number_of_death = arg[5 + 3 * ssize:5 + 4 * ssize]
    fraction_of_eligible = arg[5 + 4 * ssize:5 + 5 * ssize]
    alpha_1_1, cv1,  cv_bar01, eta1, s_up1, s_f1,  C_i_1, sigma_1, rate_1  = parameters

    sol = integrate.solve_ivp(dot_fraction_of_vaccinated, [time_ref[0], time_ref[-1]], (
                            0, 0), args=(alpha_1_1, cv1,  cv_bar01, eta1, s_up1, s_f1, C_i_1,
                                                                             sigma_1, rate_1, *arg_N, *time_ref,
                                                                             *number_of_case, *number_of_death,
                                                                             *fraction_of_eligible), t_eval=time_ref,
                              dense_output=True,method=method0)
    model_output =N*(sol.y[0, :] + sol.y[1, :])
    model_output_shifted = np.insert(model_output, 0, 0)
    model_output_shifted = np.delete(model_output_shifted, [-1])
    ee = np.asarray(model_output - model_output_shifted)
    new_vaccinated = np.asarray(new_vaccinated)
    residuals_22 = ee - (np.asarray(new_vaccinated))
    RSS = np.sum(residuals_22 ** 2)
    return RSS



def optm(arg):
    i = arg
    Ref_Time, indicator, indicator1 = [], [], []
    total_vaccinated = (OWID.loc[OWID['Location'] == i, 'Admin_Dose_1_Cumulative']).to_numpy(dtype=np.float64)
    total_shifted = np.insert(total_vaccinated, 0, 0)
    total_shifted = np.delete(total_shifted, [-1])
    new_vaccinated = total_vaccinated - total_shifted #weekly new vaccinated people
    number_of_case = OWID.loc[OWID['Location'] == i, 'new_case'].to_numpy(dtype=np.float64)
    number_of_death = OWID.loc[OWID['Location'] == i, 'new_death'].to_numpy(dtype=np.float64)
    pop_of_eligible1 = OWID.loc[OWID['Location'] == i, 'PopElig'].to_numpy(dtype=np.float64)
    pop_of_eligible = shift(pop_of_eligible1, -1, mode='nearest') # to match the week of distribution and vaccination
    N = OWID.loc[OWID['Location'] == i, 'population'].iloc[0]
    Nf = OWID.loc[OWID['Location'] == i, 'popf16'].iloc[0]
    cv_cof = fitPow.loc[fitPow['Location'] == i, 'cv_cof'].iloc[0]
    NonHesitant = N - Nf*hesitancy.loc[hesitancy['Location'] == i, 'Estimated_Strongly_Hesitant'].iloc[0]
    fraction_of_eligible = np.clip(pop_of_eligible/N, 0, NonHesitant/N) #capping the number of potential vaccinatiors
    # to number of non-hesitant individuals assuming that no vaccine doses would be wasted
    Ref_Time_2 = list(OWID.loc[OWID['Location'] == i, 'Date'])
    announcement_date = mandate.loc[mandate['Location'] == i, 'announcement'].iloc[0]
    for k in Ref_Time_2:
        ManDate = pd.to_datetime(announcement_date)
        SideDate = pd.to_datetime('01/01/2021')
        if k > ManDate:
            indc = 1 #indicated the earliest date from when mandate policies were announced
        else:
            indc = 0
        if k > SideDate:
            indc1 = 1 #indicates the earliest data on perceived risk of side effect
        else:
            indc1 = 0
        indicator.append(indc)
        indicator1.append(indc1)
        Ref_Time.append((k - Ref_Time_2[0]).days)
    time_ref = np.divide(np.array(Ref_Time, dtype=np.float64), 7) #convert days to weeks
    inArg = np.argmax(np.asarray(indicator)) #indicated the earliest datapoint from when mandate policies were announced
    inArg1 = np.argmax(np.asarray(indicator1)) #indicates the earliest datapoint on perceived risk of side effect
    np.random.seed(765) # setting seed for random function to make the inital guess on parameters retrievable
    cv_g = random.random()
    cv_bar0_g = max_cbar0*random.random()
    eta_g = min_eta + (max_eta - min_eta)*random.random()
    s_up_g = min_s + (max_s -min_s)*random.random()
    s_f_g = min_s_f + (max_s_f - min_s_f) * random.random()
    C_I_g = 0 + max_C_I * random.random()
    sigma_g = 0 + max_sigma * random.random()
    rate_g = 0 + max_rate * random.random()
    alpha_g = 0 + random.random()
    seed0 = 27 # setting seed for random function to make the optimization retrievable
    int_guess = np.array([alpha_g, cv_g,  cv_bar0_g, eta_g, s_up_g, s_f_g,  C_I_g,  sigma_g, rate_g])
    limit_list = [(0, 1), (0,1),(0, max_cbar0), (min_eta, max_eta), (min_s, max_s), (min_s_f, max_s_f),
                  (0, max_C_I), (0, max_sigma), (0, max_rate)]
    startTime = time.time()
    res_opt = optimize.dual_annealing(obj_ode, bounds=limit_list, x0=int_guess,  maxiter=
        iter_interval, seed=random.default_rng(seed=seed0), initial_temp= initalTemp, args=[len(time_ref), N, inArg,
                                                                                            inArg1, cv_cof, *time_ref,
                                                                                      *new_vaccinated,
                                                              *number_of_case, *number_of_death,
                                                              *fraction_of_eligible])

# Extracting the estimated parameters
    alpha_1, cv,  cv_bar0, eta, s_up, s_f, C_i,  sigma, rate = res_opt.x
    alpha = alpha_1*NonHesitant/N
    RSE_1 = res_opt.fun
    with open(mainPath + resultPath  +'estimated.txt', "a") as text_file:
            print(f"{i},alpha: {alpha}, alpha_1: {alpha_1},cv: {cv},  cv_bar0: {cv_bar0}, eta: {eta}, s_up: {s_up},"
                  f" s_f: {s_f}, "
                  f" C_i: {C_i},  "
                  f"sigma: {sigma},rate: {rate},"
                  , file=text_file)
    # to calculate the best estimated values
    arg_N = [N, len(time_ref), inArg, inArg1, cv_cof]
    sol = integrate.solve_ivp(dot_fraction_of_vaccinated, [time_ref[0], time_ref[-1]], (
        0, 0), args=(alpha_1, cv, cv_bar0, eta, s_up, s_f, C_i,sigma, rate, *arg_N, *time_ref,
                     *number_of_case, *number_of_death, *fraction_of_eligible), t_eval=time_ref,
                              dense_output=True, method=method0)
    model_output = N * (sol.y[0, :] + sol.y[1, :])
    model_output_shifted = np.insert(model_output, 0, 0)
    model_output_shifted = np.delete(model_output_shifted, [-1])
    new_vaccinated_fitted = np.asarray(model_output - model_output_shifted)
    print(f'new_vaccinated_fitted:{new_vaccinated_fitted.shape}')
    print(f'new_vaccinated:{new_vaccinated.shape}')
    print(f'time_ref:{time_ref.shape}')
    simulatedFile=pd.DataFrame(np.reshape(np.asarray([i]*len(time_ref)),(len(time_ref),1)),
                               columns=['Location'])
    simulatedFile['new_vaccinated_fitted'] =np.reshape(new_vaccinated_fitted, (len(time_ref),1))
    simulatedFile['new_vaccinated'] =np.reshape(new_vaccinated, (len(time_ref),1))
    simulatedFile['time_ref'] =np.reshape(time_ref, (len(time_ref),1))

    data = np.column_stack([new_vaccinated_fitted, new_vaccinated, time_ref])
    filepath = Path(mainPath + resultPath + i+'_SimFile.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    simulatedFile.to_csv(filepath)
    # datafile_path = mainPath + resultPath  + str(i) + '.txt'
    # np.savetxt(datafile_path, data, fmt=['%d', '%d', '%d'])

    return i, alpha, alpha_1, cv, cv_bar0, eta, s_up, s_f, C_i, sigma, rate


def main():

    pool_list = list()
    Number_of_cpus = multiprocessing.cpu_count()

    for p2 in StateName:
        pool_list.append(p2)

    with Pool(processes=Number_of_cpus) as run_pool:
        parallel_output = run_pool.map(optm, pool_list)
        run_pool.close()
        run_pool.join()
    f = parallel_output
    ddf = pd.DataFrame(f, columns=['code', 'alpha', 'alpha_1', 'cv', 'cv_bar0', 'eta', 's_up', 's_f', 'C_i',
                                   'sigma', 'rate'])

    filepath = Path(mainPath + resultPath +'_estimated.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ddf.to_csv(filepath)


if __name__ == '__main__':
    main()


