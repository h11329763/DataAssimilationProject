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

x_a_save = np.genfromtxt(PRojectPath+'OI/x_a_e0_0.1_1221.txt')
x_a_save_e2 = np.genfromtxt(PRojectPath+'OI/x_a_e2_0.1_1221.txt')
x_a_save_g2 = np.genfromtxt(PRojectPath+'OI/x_a_g2_0.1_1221.txt')
x_a_save_g2d2 = np.genfromtxt(PRojectPath+'OI/x_a_g2d2_0.1_1221.txt')
x_a_save_g2d3 = np.genfromtxt(PRojectPath+'OI/x_a_g2d3_0.1_1221.txt')
x_a_save_g2d4 = np.genfromtxt(PRojectPath+'OI/x_a_g2d4_0.1_1221.txt')
x_a_save_g2d8 = np.genfromtxt(PRojectPath+'OI/x_a_g2d8_0.1_1221.txt')

x_a_save_z1e0 = np.genfromtxt(PRojectPath+'OI/x_a_z1e0_0.1_1221.txt')


Bias_A_noDA = Bias_X_ts(Err_X(x_a_save_noDA, x_t_save))

Bias_e0 = Bias_X_ts(Err_X(x_a_save, x_t_save))
Bias_e2 = Bias_X_ts(Err_X(x_a_save_e2, x_t_save))
Bias_g2 = Bias_X_ts(Err_X(x_a_save_g2, x_t_save))
Bias_g2d2 = Bias_X_ts(Err_X(x_a_save_g2d2, x_t_save))
Bias_g2d3 = Bias_X_ts(Err_X(x_a_save_g2d3, x_t_save))
Bias_g2d4 = Bias_X_ts(Err_X(x_a_save_g2d4, x_t_save))
Bias_g2d8 = Bias_X_ts(Err_X(x_a_save_g2d8, x_t_save))



Err_noDA = Err_X_ts(Err_X(x_a_save_noDA, x_t_save))
Err_e0 = Err_X_ts(Err_X(x_a_save, x_t_save))
Err_e2 = Err_X_ts(Err_X(x_a_save_e2, x_t_save))
Err_g2 = Err_X_ts(Err_X(x_a_save_g2, x_t_save))
Err_g2d2 = Err_X_ts(Err_X(x_a_save_g2d2, x_t_save))
Err_g2d3 = Err_X_ts(Err_X(x_a_save_g2d3, x_t_save))
Err_g2d4 = Err_X_ts(Err_X(x_a_save_g2d4, x_t_save))
Err_g2d8 = Err_X_ts(Err_X(x_a_save_g2d8, x_t_save))
Err_z1e0 = Err_X_ts(Err_X(x_a_save_z1e0, x_t_save))


plt.clf()

plt.subplots(figsize = (11,7))

plt.plot(np.arange(nT+1) * dT, Err_noDA, color = 'black')
plt.plot(np.arange(nT+1) * dT, Err_e0, color = 'red')
plt.plot(np.arange(nT+1) * dT, Err_e2, color = 'orange')
plt.plot(np.arange(nT+1) * dT, Err_g2, color = 'olive')
plt.plot(np.arange(nT+1) * dT, Err_g2d2, color = 'green')
plt.plot(np.arange(nT+1) * dT, Err_g2d3, color = 'dodgerblue')
plt.plot(np.arange(nT+1) * dT, Err_g2d4, color = 'blue')
plt.plot(np.arange(nT+1) * dT, Err_g2d8, color = 'purple')

plt.plot(np.arange(nT+1) * dT, Err_z1e0, color = 'brown')
'''
#plt.plot(np.arange(nT+1) * dT, Bias_A_noDA_ts, color = 'black', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e0, color = 'red', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2, color = 'orange', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_g2, color = 'olive', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_g2d2, color = 'green', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_g2d3, color = 'dodgerblue', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_g2d4, color = 'blue', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_g2d8, color = 'purple', linestyle = 'dashed')
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
plt.xlim(0,10)
plt.ylim(-0.5,5)

#'''

'''
plt.xlim(0,2)
plt.ylim(-0.05,2)
#'''
plt.hlines(0.,0,100, color = 'black')

plt.legend(['No DA','Full 40','EO 20','20St1','20St2','20St3','20St4','20St8','40G1'], fontsize = 20, loc = 'upper right', ncol = 3)
plt.title('RMS Error Evolution v.s. Stagger Obs. - OI', fontsize = 25)

plt.savefig('rmse_ts_obs_nums_OI_ts_0.1_1221_0.eps', format='eps')

#plt.savefig('rmse_ts_obs_nums_OI_ts_0.1_1221_2.eps', format='eps')

#plt.savefig('rmse_ts_obs_nums_OI_ts_0.1_1221_2.png', dpi = 200)
plt.show()
#plt.xlim(0,1)
#'''
###############################################################################