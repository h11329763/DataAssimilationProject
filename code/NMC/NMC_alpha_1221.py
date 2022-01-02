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
e2_2 = np.genfromtxt(PRojectPath+'OI/x_a_e2_0.1_1221_0.2.txt')
e2_3 = np.genfromtxt(PRojectPath+'OI/x_a_e2_0.1_1221_0.3.txt')
e2_4 = np.genfromtxt(PRojectPath+'OI/x_a_e2_0.1_1221_0.4.txt')
e2_5 = np.genfromtxt(PRojectPath+'OI/x_a_e2_0.1_1221_0.5.txt')
e2_6 = np.genfromtxt(PRojectPath+'OI/x_a_e2_0.1_1221_0.6.txt')
e2_8 = np.genfromtxt(PRojectPath+'OI/x_a_e2_0.1_1221_0.8.txt')



Bias_A_noDA = Bias_X_ts(Err_X(x_a_save_noDA, x_t_save))

Bias_e0 = Bias_X_ts(Err_X(x_a_save, x_t_save))
Bias_e2_2 = Bias_X_ts(Err_X(e2_2, x_t_save))
Bias_e2_3 = Bias_X_ts(Err_X(e2_3, x_t_save))
Bias_e2_4 = Bias_X_ts(Err_X(e2_4, x_t_save))
Bias_e2_5 = Bias_X_ts(Err_X(e2_5, x_t_save))
Bias_e2_6 = Bias_X_ts(Err_X(e2_6, x_t_save))
Bias_e2_8 = Bias_X_ts(Err_X(e2_8, x_t_save))


Err_noDA = Err_X_ts(Err_X(x_a_save_noDA, x_t_save))
Err_e0 = Err_X_ts(Err_X(x_a_save, x_t_save))

Err_2 = Err_X_ts(Err_X(e2_2, x_t_save))
Err_3 = Err_X_ts(Err_X(e2_3, x_t_save))
Err_4 = Err_X_ts(Err_X(e2_4, x_t_save))
Err_5 = Err_X_ts(Err_X(e2_5, x_t_save))
Err_6 = Err_X_ts(Err_X(e2_6, x_t_save))
Err_8 = Err_X_ts(Err_X(e2_8, x_t_save))



plt.clf()

plt.subplots(figsize = (11,7))

plt.plot(np.arange(nT+1) * dT, Err_noDA, color = 'black')
plt.plot(np.arange(nT+1) * dT, Err_e0, color = 'red')
plt.plot(np.arange(nT+1) * dT, Err_2, color = 'orange')
plt.plot(np.arange(nT+1) * dT, Err_3, color = 'olive')
plt.plot(np.arange(nT+1) * dT, Err_4, color = 'green')
plt.plot(np.arange(nT+1) * dT, Err_5, color = 'dodgerblue')
plt.plot(np.arange(nT+1) * dT, Err_6, color = 'blue')
plt.plot(np.arange(nT+1) * dT, Err_8, color = 'purple')


#plt.plot(np.arange(nT+1) * dT, Bias_A_noDA_ts, color = 'black', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e0, color = 'red', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_2, color = 'orange', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_3, color = 'olive', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_4, color = 'green', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_5, color = 'dodgerblue', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_6, color = 'blue', linestyle = 'dashed')
plt.plot(np.arange(nT+1) * dT, Bias_e2_8, color = 'purple', linestyle = 'dashed')


plt.xlabel('Time [day]', fontsize = 20)
plt.ylabel('RMS Error', fontsize = 20)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.grid()

'''
plt.xlim(0,50)
plt.ylim(-1,10)

'''
plt.xlim(0,20)
plt.ylim(-0.2,6)
#'''
plt.hlines(0.1,0,20, color = 'black', linestyle = 'dotted')

plt.legend(['No DA','Full Obs',r'$ \alpha$ = 0.2',r'$ \alpha$ = 0.3',r'$ \alpha$ = 0.4',r'$ \alpha$ = 0.5',r'$ \alpha$ = 0.6',r' $\alpha$ = 0.8'], fontsize = 20, loc = 'upper right', ncol = 3)
plt.title(r'Error Evolution - EO 20, NMC Method $\alpha$ Value', fontsize = 25)
plt.savefig('NMC_Alpha_1221_1.eps', format='eps')

#plt.savefig('NMC_Alpha_1221_1.png', dpi = 200)
plt.show()
#plt.xlim(0,1)
#'''
###############################################################################