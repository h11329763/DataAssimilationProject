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
###############################################################################
x_a_save = np.zeros((nT+1,N))

x_a_save[0,:] = x_a_init[:]

members = 10
ensemble_error = 0.1

Pertb_error = 0.1

x_ak_save = np.zeros((nT+1,N,members))
x_bk_save = np.zeros((nT+1,N,members))

for i in range(members):
    x_ak_save[0,:,i] = ensemble_error * np.random.randn(N)
    x_bk_save[0,:,i] = x_ak_save[0,:,i]
    
H = np.eye(40)

obs_N = 40

########################################
#Localization
R = 3

# localization function
f_loc = np.zeros((N, N))
loc = np.arange(int(N/2) + 1)
loc = np.append(loc, loc[::-1][1:-1])
f_loc[0, :] = loc
for k in range(N - 1):
    f_loc[k + 1, :] = np.roll(f_loc[0, :], k + 1, axis = 0)
f_loc = np.exp(-(f_loc ** 2)/(2 *(R ** 2))) 
###############################################################################

inflation = 1.2

for tt in range(1,nT+1):
    tts = tt - 1
    Ts = tts * dT  # forecast start time
    Ta = tt  * dT  # forecast end time (DA analysis time)
    print('Cycle =', tt, ', Ts =', round(Ts, 10), ', Ta =', round(Ta, 10))

    #--------------
    # forecast step
    #--------------
    for i in range(members):
        
        solver = ode(lorenz96.f).set_integrator('dopri5')
        solver.set_initial_value(x_ak_save[tts,:,i], Ts).set_f_params(F)
        solver.integrate(Ta)
        
        x_bk_save[tt,:,i] = solver.y
    
    if tt % 2 == 0:        
        #---------------
        #Analysis Step
        #---------------
    
        x_m_save = np.nanmean(x_bk_save[tt,:,:], axis = -1) #[N]
        
        PfHt = np.zeros((N,obs_N))
        HPfHt = np.zeros((obs_N,obs_N))
        
        #In case of H is linear in Lorenz96, the following lines are simplified
        for i in range(members):
            PartA = np.zeros((N,1))
            PartB = np.zeros((obs_N,1))
            
            PartA[:,0] = (x_bk_save[tt,:,i] - x_m_save)
            PartB[:,0] = H.dot(x_bk_save[tt,:,i]) - H.dot(x_m_save)
            
            #PartA[:,0] = (x_bk_save[tt,:,i] - x_m_save) * inflation
            #PartB[:,0] = H.dot(x_m_save + (x_bk_save[tt,:,i] - x_m_save) * inflation) - H.dot(x_m_save)
            
            PfHt  = PfHt + 1/(members - 1) * (PartA).dot(PartB.transpose())
            HPfHt = HPfHt + 1/(members -1) * (PartB).dot(PartB.transpose())
            
        PfHt = f_loc.dot(H.transpose()) * PfHt
        HPfHt = H.dot(f_loc).dot(H.transpose()) * HPfHt
    
        K = PfHt.dot(np.linalg.inv(HPfHt + R_e))
    
        # observation 
        y_o_e2 = x_o_save[tt,:].transpose()   
    
        for i in range(members):
            # innovation
            d = y_o_e2 + Pertb_error * np.random.randn(obs_N) - H.dot(x_bk_save[tt,:,i].transpose()) #Perturbation Error
            
            x_ak_save[tt,:,i] = x_bk_save[tt,:,i] + K.dot(d)
    
        x_m_save2 = np.nanmean(x_ak_save[tt,:,:], axis = -1) #[N]
    
        for i in range(members):
            x_ak_save[tt,:,i] =  x_m_save2 + (x_ak_save[tt,:,i] - x_m_save2) * inflation
    else:
        for i in range(members):
            x_ak_save[tt,:,i] = x_bk_save[tt,:,i]
            
    tt += 1
#'''
np.savetxt('x_a_enkf_z1e0_0.1_1221_%dn.txt'%(members), np.nanmean(x_ak_save, axis = -1))
'''
for i in range(members):
    np.savetxt('x_a_enkf_e0_1203_0.1_%dn%d.txt'%(members,i), x_ak_save[:,:,i])
    #'''