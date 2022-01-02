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

ss_ = 40

E_Err_1 = np.nanmean(Err_1[ss_:])
E_Err_2 = np.nanmean(Err_2[ss_:])
E_Err_3 = np.nanmean(Err_3[ss_:])
E_Err_5 = np.nanmean(Err_5[ss_:])
E_Err_8 = np.nanmean(Err_8[ss_:])
E_Err_10 = np.nanmean(Err_10[ss_:])


#labels = ['OI/\n3D-Var', 'EKF', '4D-Var', 'EnSRF', 'Hybrid\n3DEnVar', 'Hybrid\n4D-Var']
labels = ['Localization L Value']
x = np.arange(len(labels))
width = 0.1

plt.clf()

fig, ax = plt.subplots(figsize = (5., 7))
rects1 = ax.bar(x - width/3, E_Err_1, width/8, label = 'L = 1', edgecolor = 'k', linewidth = 2)
rects2 = ax.bar(x - width/6, E_Err_2, width/8, label = 'L = 2', edgecolor = 'k', linewidth = 2)
rects3 = ax.bar(      x + 0, E_Err_3, width/8, label = 'L = 3', edgecolor = 'k', linewidth = 2)
rects4 = ax.bar(x + width/6, E_Err_5, width/8, label = 'L = 5', edgecolor = 'k', linewidth = 2)
#rects5 = ax.bar(x + width/2, E_Err_8, width/4, label = 'R = 8', edgecolor = 'k', linewidth = 2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE', fontsize = 20.)
ax.set_title('RMSE in Steady State', fontsize = 22.)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(labelsize = 20.)
ax.set_xlim([-0.06,0.04])
ax.set_ylim([0, 0.2])
ax.legend(fontsize = 20.)
ax.grid(linestyle = '--')

# fig.tight_layout()
plt.savefig('RMSE_comp_EnKF_Local_1221.eps', format = 'eps', bbox_inches = "tight")
plt.show()
