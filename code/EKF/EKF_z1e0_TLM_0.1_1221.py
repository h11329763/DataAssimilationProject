import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *

PRojectPath = 'D:/NTUGrads/Grad1/DataAssimilation/project/1118/'

# load initial condition
x_a_init = np.genfromtxt(PRojectPath+'PreRuns/x_a_init_1221.txt')

# load observations
x_o_save = np.genfromtxt(PRojectPath+'Obs/x_o_0.1_1221.txt')
###############################################################################
R_e = np.eye(40) * 0.1 ** 2
B_e = np.genfromtxt(PRojectPath+'NMC/B_NMC_1203_0.1_1221.txt') #[40,40]
alpha = 0.4
B_e = B_e  * alpha

K = B_e.dot(np.linalg.inv(B_e + R_e))
###############################################################################
x_a_save = np.zeros((nT+1,N))
x_b_save = np.zeros((nT+1,N))

Pb_save  = np.zeros((nT+1,N,N))

x_a_save[0,:] = x_a_init[:]#[0,:]
x_b_save[0,:] = x_a_init
Pb_save[0,:,:] = B_e

H = np.eye(40)

from TLM import *

for tt in range(1,nT+1):
    tts = tt - 1
    Ts = tts * dT  # forecast start time
    Ta = tt  * dT  # forecast end time (DA analysis time)
    print('Cycle =', tt, ', Ts =', round(Ts, 10), ', Ta =', round(Ta, 10))

    #--------------
    # forecast step
    #--------------
    
    solver = ode(lorenz96.f).set_integrator('dopri5')
    solver.set_initial_value(x_a_save[tts,:], Ts).set_f_params(F)
    solver.integrate(Ta)
    
    x_b_save[tt,:] = solver.y
    
    if tt % 2 == 0:    
        #dTT = 0.1dT
        TLM_n = 10
        dTT = dT/TLM_n
    
        L      = np.eye(40)
        for dTt in range(TLM_n):
            
            solver2 = ode(lorenz96.f).set_integrator('dopri5')
            solver2.set_initial_value(x_a_save[tts,:], Ts).set_f_params(F)
            solver2.integrate(Ts + dTt * dTT)
            y_new = solver2.y
            
            L_temp = np.eye(40) + dTT * Fx(y_new)
            L      = L_temp.dot(L)     
        
        M = np.copy(L)
        Pb_save[tt,:,:] = M.dot(Pb_save[tts,:,:].dot(M.transpose()))    
        Pb_save[tt,:,:] = Pb_save[tt,:,:] * 1.3
    
        # observation 
        y_o = x_o_save[tt,:].transpose()
    
        # innovation
        y_b = x_b_save[tt,:].transpose()
        d = y_o - y_b    
    
        # OI_s
        K = Pb_save[tt,:,:].dot(np.linalg.inv(Pb_save[tt,:,:] + R_e))
        
        x_a = y_b + K.dot(d)
    
        x_a_save[tt,:] = x_a
        Pb_save[tt,:,:] = (np.eye(40) - K).dot(Pb_save[tt,:,:])
    else:
        x_a_save[tt,:] = x_b_save[tt,:]

        TLM_n = 10
        dTT = dT/TLM_n
    
        L      = np.eye(40)
        for dTt in range(TLM_n):
            
            solver2 = ode(lorenz96.f).set_integrator('dopri5')
            solver2.set_initial_value(x_a_save[tts,:], Ts).set_f_params(F)
            solver2.integrate(Ts + dTt * dTT)
            y_new = solver2.y
            
            L_temp = np.eye(40) + dTT * Fx(y_new)
            L      = L_temp.dot(L)     
        
        M = np.copy(L)
        Pb_save[tt,:,:] = M.dot(Pb_save[tts,:,:].dot(M.transpose()))    
        Pb_save[tt,:,:] = Pb_save[tt,:,:] * 1.3
        
    tt += 1
#'''
# save background and analysis data
np.savetxt('x_a_ekf_z1e0_0.1_1221.txt', x_a_save)
#'''