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

members = 10


e2_10 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.00.txt')
e2_11 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.10.txt')
e2_12 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.20.txt')
e2_13 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.30.txt')
e2_14 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.40.txt')
e2_15 = np.genfromtxt('x_a_ekf_e2_0.1_1221_1.50.txt')
e2_20 = np.genfromtxt('x_a_ekf_e2_0.1_1221_2.00.txt')



Err_e2_10 = Err_X_ts(Err_X(e2_10, x_t_save))
Err_e2_11 = Err_X_ts(Err_X(e2_11, x_t_save))
Err_e2_12 = Err_X_ts(Err_X(e2_12, x_t_save))
Err_e2_13 = Err_X_ts(Err_X(e2_13, x_t_save))
Err_e2_14 = Err_X_ts(Err_X(e2_14, x_t_save))
Err_e2_15 = Err_X_ts(Err_X(e2_15, x_t_save))
Err_e2_20 = Err_X_ts(Err_X(e2_20, x_t_save))

ss_ = 40


E_Err_e2_10 = np.nanmean(Err_e2_10[ss_:])
E_Err_e2_11 = np.nanmean(Err_e2_11[ss_:])
E_Err_e2_12 = np.nanmean(Err_e2_12[ss_:])
E_Err_e2_13 = np.nanmean(Err_e2_13[ss_:])
E_Err_e2_14 = np.nanmean(Err_e2_14[ss_:])
E_Err_e2_15 = np.nanmean(Err_e2_15[ss_:])
E_Err_e2_20 = np.nanmean(Err_e2_20[ss_:])



#labels = ['OI/\n3D-Var', 'EKF', '4D-Var', 'EnSRF', 'Hybrid\n3DEnVar', 'Hybrid\n4D-Var']
labels = ['Inflation Factor']
x = np.arange(len(labels))
width = 0.7

plt.clf()

fig, ax = plt.subplots(figsize = (5., 7))
rects1 = ax.bar(x - 3, E_Err_e2_10, width, label = r'$\rho$=1.0', edgecolor = 'k', linewidth = 2, color = 'red')
rects2 = ax.bar(x - 2, E_Err_e2_11, width, label = r'$\rho$=1.1', edgecolor = 'k', linewidth = 2, color = 'orange')
rects3 = ax.bar(x - 1, E_Err_e2_12, width, label = r'$\rho$=1.2', edgecolor = 'k', linewidth = 2, color = 'olive')
rects4 = ax.bar(      x, E_Err_e2_13, width, label =r'$\rho$=1.3', edgecolor = 'k', linewidth = 2, color = 'green')
rects5 = ax.bar(x + 1, E_Err_e2_14, width, label = r'$\rho$=1.4', edgecolor = 'k', linewidth = 2, color = 'dodgerblue')
rects5 = ax.bar(x + 2, E_Err_e2_15, width, label = r'$\rho$=1.5', edgecolor = 'k', linewidth = 2, color = 'blue')
rects5 = ax.bar(x + 3, E_Err_e2_20, width, label = r'$\rho$=2.0', edgecolor = 'k', linewidth = 2, color = 'purple')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE', fontsize = 20.)
ax.set_title('RMSE in Steady State - EKF', fontsize = 22.)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(labelsize = 20.)
ax.set_xlim([-4,4])
ax.set_ylim([0, 0.15])
ax.legend(fontsize = 18., ncol = 2)
ax.grid(linestyle = '--')

# fig.tight_layout()
plt.savefig('RMSE_comp_EK_inflationF_1221.eps', format='eps', bbox_inches = 'tight')
plt.show()
