import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *

PRojectPath = 'D:/NTUGrads/Grad1/DataAssimilation/project/1118/'

# load initial condition
x_a_init = np.genfromtxt(PRojectPath+'PreRuns/x_a_init_1221.txt')

# load observations
x_o_save = np.genfromtxt(PRojectPath+'Obs/x_o_g2_0.1_1221.txt')

#Stack only non-nan
obs_N = 20
x_o_save_e2 = np.zeros((nT + 1,obs_N))

for time in range(nT+1):
    x_o_save_e2[time,:] = x_o_save[time,:][~np.isnan(x_o_save[time,:])]

###############################################################################
R_e = np.eye(20) * 0.1 ** 2

B_e = np.genfromtxt(PRojectPath+'NMC/B_NMC_1203_0.1_1221.txt') #[40,40]
alpha = 0.4
B_e = B_e  * alpha
###############################################################################
H1 = np.zeros((obs_N,N))
H2 = np.zeros((obs_N,N))

for i in range(obs_N):
    H1[i, 2 * i] = 1
    H2[i, 2 * i + 1] = 1

K11 = B_e.dot(H1.transpose())
K12 = H1.dot(K11)
K13 = np.linalg.inv(K12 + R_e)
K1 = K11.dot(K13)

K21 = B_e.dot(H2.transpose())
K22 = H2.dot(K21)
K23 = np.linalg.inv(K22 + R_e)
K2 = K21.dot(K23)
###############################################################################
x_a_save = np.zeros((nT+1,N))
x_b_save = np.zeros((nT+1,N))

x_a_save[0,:] = x_a_init[:]#[0,:]

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
    #--------------
    # analysis step
    #--------------

    # background
    x_b = x_b_save[tt,:].transpose()

    # observation
    y_o_e2 = x_o_save_e2[tt,:].transpose()

    # innovation
    #y_b = np.dot(H, x_b)
    if tt % 2 == 0:
        H = np.copy(H1)
        K = np.copy(K1)
    else:
        H = np.copy(H2)
        K = np.copy(K2)
        
    y_b = H.dot(x_b)
    d = y_o_e2 - y_b    

    x_c = K.dot(d)

    # OI
    x_a = x_b + x_c

    x_a_save[tt,:] = x_a
    
    tt += 1
#'''
#'''
# save background and analysis data
np.savetxt('x_a_g2_0.1_1221.txt', x_a_save)
#'''