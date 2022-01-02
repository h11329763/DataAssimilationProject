import numpy as np
from settings import *
import matplotlib.pyplot as plt
from settings import *
from scipy import stats
from matplotlib import gridspec


PRojectPath = 'D:/NTUGrads/Grad1/DataAssimilation/project/1118/'

x_t_save = np.genfromtxt(PRojectPath+'PreRuns/x_t_1221.txt')



N = 40

x_o_save = np.zeros((nT+1,N), dtype='f8')

for time in range(nT+1):

    #x_b_state = x_b_save[Time,:]    
    x_t_state = x_t_save[time,:]

    x_o_state = np.copy(x_t_state)

    #Take observation  with some random noise

    sigma_o0 = 0.1  # size of initial perturpation
    x_o_state = x_o_state + sigma_o0 * np.random.randn(N)

    x_o_save[time,:] = x_o_state
    

x_o_save_e2 = np.copy(x_o_save[:])


for time in range(nT+1):
    if time % 16 < 8:
        for nn in range(20):
            x_o_save_e2[time,2 * nn + 1] = np.nan
    else:
        for nn in range(20):
            x_o_save_e2[time,2 * nn] = np.nan
            
np.savetxt('x_o_g2d8_0.1_1221.txt', x_o_save_e2)