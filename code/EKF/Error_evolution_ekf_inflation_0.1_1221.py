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
x_t_save = np.genfromtxt(PRojectPath+'PreRuns/x_t_1221.txt')

# load data
x_a_save_noDA = np.genfromtxt(PRojectPath+'OI/x_a_noDA.txt')

e0 = np.genfromtxt('x_a_ekf_e0_0.1_1221.txt')

e2_10 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.00.txt')
e2_11 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.10.txt')
e2_12 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.20.txt')
e2_13 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.30.txt')
e2_14 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.40.txt')
e2_15 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.50.txt')
e2_20 = np.genfromtxt('x_a_ekf_e2_0.1_1221_2.00.txt')


Err_noDA = Err_X_ts(Err_X(x_a_save_noDA, x_t_save))
Err_e0 = Err_X_ts(Err_X(e0, x_t_save))
Err_e2_10 = Err_X_ts(Err_X(e2_10, x_t_save))
Err_e2_11 = Err_X_ts(Err_X(e2_11, x_t_save))
Err_e2_12 = Err_X_ts(Err_X(e2_12, x_t_save))
Err_e2_13 = Err_X_ts(Err_X(e2_13, x_t_save))
Err_e2_14 = Err_X_ts(Err_X(e2_14, x_t_save))
Err_e2_15 = Err_X_ts(Err_X(e2_15, x_t_save))
Err_e2_20 = Err_X_ts(Err_X(e2_20, x_t_save))


plt.clf()

plt.subplots(figsize = (11,7))

plt.plot(np.arange(nT+1) * dT, Err_noDA, color = 'black')
plt.plot(np.arange(nT+1) * dT, Err_e0, color = 'red')
plt.plot(np.arange(nT+1) * dT, Err_e2_10, color = 'orange')
plt.plot(np.arange(nT+1) * dT, Err_e2_11, color = 'olive')
plt.plot(np.arange(nT+1) * dT, Err_e2_12, color = 'green')
plt.plot(np.arange(nT+1) * dT, Err_e2_13, color = 'dodgerblue')
plt.plot(np.arange(nT+1) * dT, Err_e2_14, color = 'blue')
plt.plot(np.arange(nT+1) * dT, Err_e2_15, color = 'purple')
plt.plot(np.arange(nT+1) * dT, Err_e2_20, color = 'pink')

'''
plt.plot(np.arange(nT+1) * dT, Bias_e0, color = 'red', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_10, color = 'orange', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_11, color = 'olive', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_12, color = 'green', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_13, color = 'dodgerblue', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_14, color = 'blue', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_15, color = 'purple', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_20, color = 'pink', linestyle = 'dashed')
#'''

plt.xlabel('Time [day]', fontsize = 20)
plt.ylabel('RMS Error', fontsize = 20)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.grid()

#'''
plt.xlim(0,20)
plt.ylim(-1,10)
#'''
'''
plt.xlim(0,2)
plt.ylim(-0.2,8)
#'''
plt.hlines(0,0,2, color = 'black')

plt.legend(['No DA','Full 40',r'$\rho$=1.0',r'$\rho$=1.1',r'$\rho$=1.2',r'$\rho$=1.3',r'$\rho$=1.4',r'$\rho$=1.5',r'$\rho$=2.0'], fontsize = 20, loc = 'upper right', ncol = 3)
plt.title('RMS Error Evolution - EO 20, Inflation in EKF', fontsize = 25)
#plt.savefig('rmse_ts_obs_nums_ekf_inflation_0.1_1221.png', dpi = 200)
plt.savefig('rmse_ts_obs_nums_ekf_inflation_0.1_1221.eps', format='eps')

plt.show()
#plt.xlim(0,1)
#'''
###############################################################################