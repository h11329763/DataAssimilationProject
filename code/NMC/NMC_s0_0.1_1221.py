"""
The data assimilation system (no assimilation example)
Load:
  x_a_init.txt
Save:
  x_b.txt
  x_a.txt
"""
import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *

PRojectPath = 'D:/NTUGrads/Grad1/DataAssimilation/project/1118/'

# load initial condition
x_a_init = np.genfromtxt(PRojectPath+'PreRuns/x_a_init_1221.txt')

# load observations
x_o_save = np.genfromtxt(PRojectPath+'Obs/x_o_0.1_1221.txt')

x_t_save = np.genfromtxt(PRojectPath+'PreRuns/x_t_1221.txt')

###############################################################################
#Compute Bg and Obs Error Covariance Matrix
R_ = np.zeros((N,N,nT+1))
#BR_t_ = np.zeros((N,N,nT))

for time in range(1,nT+1):

    x_t_state = x_t_save[time,:]

    x_o_state = x_o_save[time,:]

    e_o = np.zeros((N,1))
    e_o[:,0] = x_o_state - x_t_state

    
    R   = e_o * e_o.transpose()
    R_[:,:,time] = R
    
R_e_ = np.nanmean(R_, axis = -1)
R_e = np.zeros((N,N))

for zz in range(N):
    R_e[zz,zz] = R_e_[zz,zz]

#print(np.nanmean(B_e),np.nanmean(R_e))
B_e = np.copy(R_e)
K = B_e.dot(np.linalg.inv(B_e + R_e))
###############################################################################
x_a_save = np.zeros((nT+1,N))
x_b_save = np.zeros((nT+1,N))


x_a_save[0,:] = x_a_init[:]
#'''
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
    y_o = x_o_save[tt,:].transpose()

    # innovation
    #y_b = np.dot(H, x_b)
    y_b = x_b
    d = y_o - y_b    

    # OI
    x_a = x_b + K.dot(d)

    x_a_save[tt,:] = x_a
    
    tt += 1

# save background and analysis data
np.savetxt('x_a_BR_0.1_1221.txt', x_a_save)
#'''