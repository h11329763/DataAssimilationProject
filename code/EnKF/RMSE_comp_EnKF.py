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

# load data
e0 = np.genfromtxt('x_a_enkf_e0_0.1_1221_%dn.txt'%(members))
e2 = np.genfromtxt('x_a_enkf_e2_0.1_1221_%dn.txt'%(members))
g2 = np.genfromtxt('x_a_enkf_g2_0.1_1221_%dn.txt'%(members))
g2d2 = np.genfromtxt('x_a_enkf_g2d2_0.1_1221_%dn.txt'%(members))
g2d3 = np.genfromtxt('x_a_enkf_g2d3_0.1_1221_%dn.txt'%(members))
g2d4 = np.genfromtxt('x_a_enkf_g2d4_0.1_1221_%dn.txt'%(members))
g2d8 = np.genfromtxt('x_a_enkf_g2d8_0.1_1221_%dn.txt'%(members))
z1e0 = np.genfromtxt('x_a_enkf_z1e0_0.1_1221_%dn.txt'%(members))


Err_e0 = Err_X_ts(Err_X(e0, x_t_save))
Err_e2 = Err_X_ts(Err_X(e2, x_t_save))
Err_g2 = Err_X_ts(Err_X(g2, x_t_save))
Err_g2d2 = Err_X_ts(Err_X(g2d2, x_t_save))
Err_g2d3 = Err_X_ts(Err_X(g2d3, x_t_save))
Err_g2d4 = Err_X_ts(Err_X(g2d4, x_t_save))
Err_g2d8 = Err_X_ts(Err_X(g2d8, x_t_save))
Err_z1e0 = Err_X_ts(Err_X(z1e0, x_t_save))

ss_ = 40

E_Err_e0 = np.nanmean(Err_e0[ss_:])
E_Err_e2 = np.nanmean(Err_e2[ss_:])
E_Err_g2 = np.nanmean(Err_g2[ss_:])
E_Err_g2d2 = np.nanmean(Err_g2d2[ss_:])
E_Err_g2d3 = np.nanmean(Err_g2d3[ss_:])
E_Err_g2d4 = np.nanmean(Err_g2d4[ss_:])
E_Err_g2d8 = np.nanmean(Err_g2d8[ss_:])
E_Err_z1e0 = np.nanmean(Err_z1e0[ss_:])


#labels = ['OI/\n3D-Var', 'EKF', '4D-Var', 'EnSRF', 'Hybrid\n3DEnVar', 'Hybrid\n4D-Var']
labels = ['Stagger Method']
x = np.arange(len(labels))
width = 0.7

plt.clf()

fig, ax = plt.subplots(figsize = (5., 7))
rects1 = ax.bar(x - 3, E_Err_e0, width, label = 'Full 40', edgecolor = 'k', linewidth = 2, color = 'red')
rects2 = ax.bar(x - 2, E_Err_e2, width, label = 'EO 20', edgecolor = 'k', linewidth = 2, color = 'orange')
rects3 = ax.bar(x - 1, E_Err_g2, width, label = '20St1', edgecolor = 'k', linewidth = 2, color = 'olive')
rects4 = ax.bar(      x, E_Err_g2d2, width, label = '20St2', edgecolor = 'k', linewidth = 2, color = 'green')
rects5 = ax.bar(x + 1, E_Err_g2d3, width, label = '20St3', edgecolor = 'k', linewidth = 2, color = 'dodgerblue')
rects5 = ax.bar(x + 2, E_Err_g2d4, width, label = '20St4', edgecolor = 'k', linewidth = 2, color = 'blue')
rects5 = ax.bar(x + 3, E_Err_g2d8, width, label = '20St8', edgecolor = 'k', linewidth = 2, color = 'purple')
rects5 = ax.bar(x + 4, E_Err_z1e0, width, label = '40G1', edgecolor = 'k', linewidth = 2, color = 'brown')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE', fontsize = 20.)
ax.set_title('RMSE in Steady State - EnKF', fontsize = 22.)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(labelsize = 20.)
ax.set_xlim([-4,5])
ax.set_ylim([0, 0.15])
ax.legend(fontsize = 18., ncol = 2)
ax.grid(linestyle = '--')

# fig.tight_layout()
plt.savefig('RMSE_comp_EnKF_1221.eps', format='eps', bbox_inches = 'tight')

#plt.savefig('RMSE_comp_EnKF_1221.png', dpi = 200, bbox_inches = 'tight')
plt.show()
