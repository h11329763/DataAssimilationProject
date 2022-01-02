def Err_X(x_a_save,x_t_save):
    return(x_a_save - x_t_save)

def Bias_X_ts(Err_X):    
    return(np.nanmean(Err_X, axis = 1))

def Err_X_ts(Err_X):
    return(np.nanmean(Err_X ** 2, axis = 1) ** 0.5)

import numpy as np
from settings import *
import matplotlib.pyplot as plt

PRojectPath = 'D:/NTUGrads/Grad1/DataAssimilation/project/1118/'

# load observations
#x_o_save = np.genfromtxt(PRojectPath+'Obs/x_o_e2.txt')
x_t_save = np.genfromtxt(PRojectPath+'PreRuns/x_t_1221.txt')


# load data

x_a_save_noDA = np.genfromtxt(PRojectPath+'OI/x_a_noDA.txt')
Err_noDA = Err_X_ts(Err_X(x_a_save_noDA, x_t_save))

#x_a_save = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e0_0.1_1221_15n.txt')
e2_1 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.00_10n.txt')
e2_11 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.10_10n.txt')
e2_12 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.20_10n.txt')
e2_13 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.30_10n.txt')
e2_14 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.40_10n.txt')

#'''
Err_1 = Err_X_ts(Err_X(e2_1, x_t_save)) 
Err_11 = Err_X_ts(Err_X(e2_11, x_t_save)) 
Err_12 = Err_X_ts(Err_X(e2_12, x_t_save)) 
Err_13 = Err_X_ts(Err_X(e2_13, x_t_save)) 
Err_14 = Err_X_ts(Err_X(e2_14, x_t_save)) 

plt.clf()

plt.subplots(figsize = (11,7))

plt.plot(np.arange(nT+1) * dT, Err_noDA, color = 'black')

plt.plot(np.arange(nT+1) * dT, Err_1, color = 'red')
plt.plot(np.arange(nT+1) * dT, Err_11, color = 'orange')
plt.plot(np.arange(nT+1) * dT, Err_12, color = 'olive')
plt.plot(np.arange(nT+1) * dT, Err_13, color = 'green')
plt.plot(np.arange(nT+1) * dT, Err_14, color = 'dodgerblue')

plt.xlabel('Time [day]', fontsize = 20)
plt.ylabel('RMS Error', fontsize = 20)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.grid()

#'''
plt.xlim(0,20)
plt.ylim(-1,10)

'''
plt.xlim(0,2)
plt.ylim(-0.2,5)
#'''
plt.hlines(0,0,100, color = 'black')

plt.legend(['No DA',r'$\rho$=1.0',r'$\rho$=1.1',r'$\rho$=1.2',r'$\rho$=1.3',r'$\rho$=1.4'], fontsize = 20, loc = 'upper right', ncol = 3)

plt.title(r'Error Evolution - EO 20, EnKF Inflation Value', fontsize = 25)
plt.savefig('EnKF_inflation_1221.eps',format = 'eps')
plt.show()
#plt.xlim(0,1)
#'''
###############################################################################