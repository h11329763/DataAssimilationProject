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
x_a_save_noDA = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_NoR_10n.txt')

#x_a_save = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e0_0.1_1221_15n.txt')
e2_1 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_1R_10n.txt')
e2_2 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_2R_10n.txt')
e2_3 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_3R_10n.txt')
e2_5 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_5R_10n.txt')
e2_8 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_8R_10n.txt')
e2_10 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_10R_10n.txt')

Err_noDA = Err_X_ts(Err_X(x_a_save_noDA, x_t_save))
Err_1 = Err_X_ts(Err_X(e2_1, x_t_save)) 
Err_2 = Err_X_ts(Err_X(e2_2, x_t_save))
Err_3 = Err_X_ts(Err_X(e2_3, x_t_save))

Err_5 = Err_X_ts(Err_X(e2_5, x_t_save))
Err_8 = Err_X_ts(Err_X(e2_8, x_t_save))
Err_10 = Err_X_ts(Err_X(e2_10, x_t_save))

plt.clf()

plt.subplots(figsize = (11,7))

plt.plot(np.arange(nT+1)[:77] * dT, Err_noDA[:77], color = 'black')
plt.plot(np.arange(nT+1) * dT, Err_1, color = 'red')
plt.plot(np.arange(nT+1) * dT, Err_2, color = 'orange')
plt.plot(np.arange(nT+1) * dT, Err_3, color = 'olive')
plt.plot(np.arange(nT+1) * dT, Err_5, color = 'green')
plt.plot(np.arange(nT+1) * dT, Err_8, color = 'dodgerblue')
plt.plot(np.arange(nT+1) * dT, Err_10, color = 'blue')
#plt.plot(np.arange(nT+1) * dT, Err_8, color = 'purple')

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

plt.legend(['No Local.','L = 1','L = 2','L = 3','L = 5','L = 8','L = 10'], fontsize = 20, loc = 'upper right', ncol = 3)
plt.title(r'Error Evolution - EO 20, EnKF Localization L Value', fontsize = 25)
plt.savefig('EnKF_Local_1221.eps', format = 'eps')
plt.show()
#plt.xlim(0,1)
#'''
###############################################################################