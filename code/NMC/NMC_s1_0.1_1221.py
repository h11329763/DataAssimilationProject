import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *
import matplotlib.pyplot as plt


PRojectPath = 'D:/NTUGrads/Grad1/DataAssimilation/project/1118/'

# load initial condition
x_a_init = np.genfromtxt(PRojectPath+'PreRuns/x_a_init_1221.txt')

# load observations
x_o_save = np.genfromtxt(PRojectPath+'Obs/x_o_0.1_1221.txt')

x_c_save = np.genfromtxt(PRojectPath+'NMC/x_a_BR_0.1_1221.txt')
###############################################################################

###############################################################################
NMC_ = np.zeros((N,N))

NMC_total = np.zeros((nT,N,N))
NMC_N = 900

for NMC_t in range( 100, NMC_N + 1):
    x_c1_save = np.zeros((nT+1,N)) 
    x_c2_save = np.zeros((nT+1,N))
    
    x_c1_save[NMC_t,:]   =  x_c_save[NMC_t,:]
    x_c2_save[NMC_t + 4,:] =  x_c_save[NMC_t + 4,:]

#'''    
    for tt in range(NMC_t + 1, NMC_t + 9):
        tts = tt - 1
        Ts = tts * dT  # forecast start time
        Ta = tt  * dT  # forecast end time (DA analysis time)
        #print('Cycle =', tt, ', Ts =', round(Ts, 10), ', Ta =', round(Ta, 10))
        
        # forecast step
        solver = ode(lorenz96.f).set_integrator('dopri5')
        solver.set_initial_value(x_c1_save[tts,:], Ts).set_f_params(F)
        solver.integrate(Ta)       
        
        x_c1_save[tt,:] = solver.y

        tt += 1
#'''       
    for tt in range( NMC_t + 5, NMC_t + 9):
        tts = tt - 1
        Ts = tts * dT  # forecast start time
        Ta = tt  * dT  # forecast end time (DA analysis time)
        #print('Cycle =', tt, ', Ts =', round(Ts, 10), ', Ta =', round(Ta, 10))
    
        # forecast step
        solver = ode(lorenz96.f).set_integrator('dopri5')
        solver.set_initial_value(x_c2_save[tts,:], Ts).set_f_params(F)
        solver.integrate(Ta)
        
        x_c2_save[tt,:] = solver.y

        tt += 1
#'''        
    NMC = np.zeros((N,1))
    NMC[:,0] =  x_c2_save[NMC_t + 8,:] - x_c1_save[NMC_t + 8,:]

    NMC_e = NMC * NMC.transpose()    
    NMC_total[NMC_t + 8,:,:] = NMC_e

    NMC_dia = np.diag(NMC_e)
    print(np.nanmean(NMC_dia))
    print(NMC_t)
#'''
#'''
###############################################################################
NMC_ = np.nansum(NMC_total, axis = 0)/(NMC_N - 100 - 1)

# save NMC Matrix
np.savetxt('NMC_TEMP_0.1_1221.txt', NMC_)
#np.savetxt('NMC_TEMP_0.1.txt', NMC_)
###############################################################################
#'''
'''
X = np.arange(0.5,41.5,1)
X_ticks = np.arange(1,40.1,1)

plt.clf()
plt.subplots(figsize = (8,7))
plt.pcolormesh( X, X, NMC_, vmin = -1, vmax = 1, cmap = 'jet')
plt.colorbar()

plt.xlim(0.5,40.5)
plt.ylim(0.5,40.5)

plt.xticks(X_ticks, fontsize = 7)
plt.yticks(X_ticks, fontsize = 7)

plt.grid()
plt.show()
#'''
