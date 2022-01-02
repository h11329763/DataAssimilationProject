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
e2_1 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.00_10n.txt')
e2_11 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.10_10n.txt')
e2_12 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.20_10n.txt')
e2_13 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.30_10n.txt')
e2_14 = np.genfromtxt(PRojectPath+'EnKF/x_a_enkf_e2_0.1_1221_infla1.40_10n.txt')


Err_noDA = Err_X_ts(Err_X(x_a_save_noDA, x_t_save))
Err_1 = Err_X_ts(Err_X(e2_1, x_t_save)) 
Err_11 = Err_X_ts(Err_X(e2_11, x_t_save))
Err_12 = Err_X_ts(Err_X(e2_12, x_t_save))
Err_13 = Err_X_ts(Err_X(e2_13, x_t_save))
Err_14 = Err_X_ts(Err_X(e2_14, x_t_save))

ss_ = 40

E_Err_1 = np.nanmean(Err_1[ss_:])
E_Err_11 = np.nanmean(Err_11[ss_:])
E_Err_12 = np.nanmean(Err_12[ss_:])
E_Err_13 = np.nanmean(Err_13[ss_:])
E_Err_14 = np.nanmean(Err_14[ss_:])


#labels = ['OI/\n3D-Var', 'EKF', '4D-Var', 'EnSRF', 'Hybrid\n3DEnVar', 'Hybrid\n4D-Var']
labels = [r'Inflation $\rho$  Value']
x = np.arange(len(labels))
width = 0.1

plt.clf()

fig, ax = plt.subplots(figsize = (5., 7))
rects0 = ax.bar(x - width/3, E_Err_1, width/8, label = r'$\rho$=1.0', edgecolor = 'k', linewidth = 2)
rects1 = ax.bar(x - width/6, E_Err_11, width/8, label = r'$\rho$=1.1', edgecolor = 'k', linewidth = 2)
rects2 = ax.bar(x          , E_Err_12, width/8, label = r'$\rho$=1.2', edgecolor = 'k', linewidth = 2)
rects3 = ax.bar(x + width/6, E_Err_13, width/8, label = r'$\rho$=1.3', edgecolor = 'k', linewidth = 2)
rects4 = ax.bar(x + width/3, E_Err_14, width/8, label = r'$\rho$=1.4', edgecolor = 'k', linewidth = 2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE', fontsize = 20.)
ax.set_title('RMSE in Steady State', fontsize = 22.)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(labelsize = 20.)
ax.set_xlim([-0.06,0.06])
ax.set_ylim([0, 0.5])
ax.legend(fontsize = 20.)
ax.grid(linestyle = '--')

plt.savefig('RMSE_comp_EnKF_inflation_1221.eps',format = 'eps', bbox_inches = "tight")
plt.show()
