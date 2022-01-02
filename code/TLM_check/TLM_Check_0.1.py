import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *

import matplotlib.pyplot as plt

PRojectPath = 'D:/NTUGrads/Grad1/DataAssimilation/project/1118/'
###############################################################################
x_c1_save = np.zeros((nT+1,N))
x_c2_save = np.zeros((nT+1,N))

dx0 = np.zeros(N)

dx0[20] = 0.5

x_c2_save[0,:] = dx0#x_a_init[:] + dx0

NonLinErr = np.zeros((nT+1,N))
LinErr    = np.zeros((nT+1,N))

NonLinErr[0,:] = dx0
LinErr[0,:] = dx0
#'''
###############################################################################
XX = np.arange(0,40,1)

#'''
if 1 == 1:
    
    plt.clf()
    plt.subplots(figsize = (9,8))
    
    plt.plot(XX, dx0, color = 'darkgray')
    plt.plot(XX, LinErr[0,:], color = 'black', marker = 'o')
    plt.plot(XX, NonLinErr[0,:], color = 'red', marker = 's')

    plt.legend([r'$\Delta X_{0}$',r'$\Delta X$ [linear]',r'$\delta X$ [Non-linear]'], fontsize = 20, loc = 'upper right')
    plt.xlim(0,40)
    plt.ylim(-3,4)
    plt.grid()
    
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    plt.xlabel('X', fontsize = 20)
    plt.ylabel('Error', fontsize = 20)
    
    plt.text(4, 3.55, r'$|\Delta X$| = %8.6f'%((np.nansum(LinErr[0,:] ** 2)) ** 0.5), fontsize = 20)
    plt.text(4, 3.15, r'$|\delta X$| = %8.6f'%((np.nansum(NonLinErr[0,:] ** 2)) ** 0.5), fontsize = 20)
    plt.text(2, 2.5, r'$|\Delta X$|/$|\delta X$| = %8.6f'%((np.nansum(LinErr[0,:] ** 2)) ** 0.5/(np.nansum(NonLinErr[0,:] ** 2)) ** 0.5), fontsize = 20)

    plt.title('Linear v.s. Nonlinear Growth, $\Delta t$ = %4.2f'%0, fontsize = 25)

    plt.savefig('TLM_Check_%4.2f.png'%0)#, bbox_inches = 'tight')
    plt.show()

from TLM import *


for tt in range(1,31):#nT+1):
    tts = tt - 1
    Ts = tts * dT  # forecast start time
    Ta = tt  * dT  # forecast end time (DA analysis time)
    print('Cycle =', tt, ', Ts =', round(Ts, 10), ', Ta =', round(Ta, 10))

    #--------------
    # forecast step
    #--------------
    
    solver = ode(lorenz96.f).set_integrator('dopri5')
    solver.set_initial_value(x_c1_save[tts,:], Ts).set_f_params(F)
    solver.integrate(Ta)
    
    x_c1_save[tt,:] = solver.y

    solver = ode(lorenz96.f).set_integrator('dopri5')
    solver.set_initial_value(x_c2_save[tts,:], Ts).set_f_params(F)
    solver.integrate(Ta)
    
    x_c2_save[tt,:] = solver.y
    
    #Non Linear Error    
    NonLinErr[tt,:] = x_c2_save[tt,:] - x_c1_save[tt,:]
    
    #Linear Error
    #dTT = 0.1dT
    TLM_n = 10
    dTT = dT/TLM_n

    L      = np.eye(40)
    for dTt in range(TLM_n):
        
        solver2 = ode(lorenz96.f).set_integrator('dopri5')
        solver2.set_initial_value(x_c1_save[tts,:], Ts).set_f_params(F)
        solver2.integrate(Ts + dTt * dTT)
        y_new = solver2.y
        
        L_temp = np.eye(40) + dTT * Fx(y_new)
        L      = L_temp.dot(L)     
    
    M = np.copy(L)
    
    LinErr[tt,:] = M.dot(LinErr[tts,:])
    ###########################################################################
    plt.clf()
    plt.subplots(figsize = (9,8))
    
    plt.plot(XX, dx0, color = 'darkgray')
    plt.plot(XX, LinErr[tt,:], color = 'black', marker = 'o')
    plt.plot(XX, NonLinErr[tt,:], color = 'red', marker = 's')
    
    plt.legend([r'$\Delta X_{0}$',r'$\Delta X$ [linear]',r'$\delta X$ [Non-linear]'], fontsize = 20, loc = 'upper right')
    plt.xlim(0,40)
    plt.ylim(-3,4)
    plt.grid()
    
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    plt.xlabel('X', fontsize = 20)
    plt.ylabel('Error', fontsize = 20)
    
    plt.text(4, 3.55, r'$|\Delta X$| = %8.6f'%((np.nansum(LinErr[tt,:] ** 2)) ** 0.5), fontsize = 20)
    plt.text(4, 3.15, r'$|\delta X$| = %8.6f'%((np.nansum(NonLinErr[tt,:] ** 2)) ** 0.5), fontsize = 20)
    plt.text(2, 2.5, r'$|\Delta X$|/$|\delta X$| = %8.6f'%((np.nansum(LinErr[tt,:] ** 2)) ** 0.5/(np.nansum(NonLinErr[tt,:] ** 2)) ** 0.5), fontsize = 20)

    plt.title('Linear v.s. Nonlinear Growth, $\Delta t$ = %4.2f'%Ta, fontsize = 25)
    
    plt.savefig('TLM_Check_%4.2f.png'%Ta)#, bbox_inches = 'tight')
    plt.show()
    ###########################################################################
    tt += 1
