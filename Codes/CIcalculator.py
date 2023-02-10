import pandas as pd
from pathlib import Path
import numpy as np
import re
import os




def percentile(FilePath, alpha_est, order, up_prcnt, lo_prcnt):
    alpha = []
    counter = 0
    with open(FilePath, 'r') as fileOutput:
        readLines = fileOutput.readlines()
        for line in readLines:
            search_res = re.search(ptrn, "".join(line.split()))
            if search_res is None:
                continue
            if float(search_res.group(3)) < 0:
                print('Nothing')
                continue
            counter = counter + 1
            alphaDummy = float(search_res.group(order))
            alpha.append(alphaDummy)
    alpha.append(alpha_est)
    alphaP_CI_l = np.percentile(alpha, lo_prcnt)
    alphaP_CI_u = np.percentile(alpha, up_prcnt)
    return alphaP_CI_l, alphaP_CI_u, counter #


#ptrn contains the pattern according to which the estimated values obtained using synthesized data stored
ptrn ='Code:(.*),alpha:(.*),alpha_1:(.*),cv:(.*),cv_bar0:(.*),eta:(.*),s_up:(.*),s_f:(.*),C_i:(.*),sigma:(.*),rate:(.*)'
dirname = os.path.dirname(__file__)
Path_file = dirname + '/Result/BootResult/'
fitFile = dirname + '/Result/fitFile.csv'
# The name pattern of the text files containing the estimated values obtained using synthesized data stored
startName = 'non'
endName = '.txt'
ptrn1 = 'non_(.*).txt'
df = pd.read_csv(fitFile)
results = pd.DataFrame( columns=['code', 'NbootData', 'parameter', 'EstValue', 'CI-', 'CI+', 'ConfidenceInterval'])
directory = os.fsencode(Path_file )

for file in os.listdir(directory): # searching the file containing the estimated values
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        if filename.startswith(startName) and filename.endswith(endName):
            search_res = re.search(ptrn1, "".join(filename.split()))
            code = search_res.group(1)
            iter_interval = int(startName)

        else:
            continue
        up_prcnt = 97.5 # upper percentile
        lo_prcnt = 2.5 # lower percentile

        pName = 'alpha'
        p_estimated = (df.loc[(df['Location'] == code)][pName]).values[0]
        CI_l, CI_u,  counter= percentile(Path_file + filename, p_estimated, 2, up_prcnt, lo_prcnt)

        dfr = pd.DataFrame([[code, counter, pName, p_estimated,
                         CI_l, CI_u, CI_u - CI_l]],
                       columns=['code', 'NbootData', 'parameter',  'EstValue', 'CI-', 'CI+', 'ConfidenceInterval'])
        results = pd.concat([results, dfr])
        pName = 'cv_bar0'
        p_estimated = (df.loc[(df['Location'] == code) ][pName]).values[0]
        CI_l, CI_u,  counter = percentile(Path_file + filename, p_estimated, 5, up_prcnt, lo_prcnt)
        dfr = pd.DataFrame([[code, counter, pName, p_estimated,
                             CI_l, CI_u, CI_u - CI_l]],
                           columns=['code', 'NbootData', 'parameter', 'EstValue', 'CI-', 'CI+', 'ConfidenceInterval'])
        results = pd.concat([results, dfr])

filepath1 = Path(Path_file + 'CI.csv')
filepath1.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(filepath1)






