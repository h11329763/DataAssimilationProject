import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

import lingcolor_v2_r

# load initial condition
NMC = np.genfromtxt('NMC_TEMP_0.1_1221.txt')

#for i in range(40):
#    print(NMC[i,i])
NMC_dia = np.diag(NMC)
print(np.nanmean(NMC_dia))

#NMC = NMC/(np.nanmean(NMC_dia)/(0.1 ** 2))

#'''
NMC21 = np.zeros(21)
NMC21count = np.zeros(21)

for i in range(40):
    for j in range(40):
        dis = np.abs(i-j)
        
        if dis < 21:
            XD = 1
        else:
            dis = 40 - dis
        
        NMC21[dis] = NMC21[dis] + NMC[i,j]
        NMC21count[dis] = NMC21count[dis] + 1

    print(i)
 
NMC21 = NMC21/NMC21count   

NMC_new = np.zeros((40,40))

for i in range(40):
    for j in range(40):
        dis = np.abs(i-j)
        
        if dis < 21:
            XD = 1
        else:
            dis = 40 - dis   
            
        NMC_new[i,j] = NMC21[dis]


#np.savetxt('B_NMC_1203_0.1_1221.txt', NMC_new)
        
###############################################################################

X = np.arange(0.5,41.5,1)
X_ticks = np.arange(1,40.1,1)
plt.clf()
plt.subplots(figsize = (8,7))

plt.pcolormesh( X, X, NMC, vmin = -0.01 , vmax = 0.01, cmap = 'lingcolor_v2_r')

plt.colorbar()

plt.xlim(0.5,40.5)
plt.ylim(0.5,40.5)

plt.xticks(X_ticks, fontsize = 7)
plt.yticks(X_ticks, fontsize = 7)

plt.grid()

plt.title('Matrix $P^{b}$ - NMC Method Temp', fontsize = 25)
plt.savefig('B_NMC_temp_1203_0.1_1221.eps', format='eps')

#plt.savefig('B_NMC_temp_1203_0.1_1221.png', dpi = 200)
plt.show()

plt.clf()
plt.subplots(figsize = (8,7))

plt.pcolormesh( X, X, NMC_new, vmin = -0.01 , vmax = 0.01, cmap = 'lingcolor_v2_r')
plt.colorbar()

plt.xlim(0.5,40.5)
plt.ylim(0.5,40.5)

plt.xticks(X_ticks, fontsize = 7)
plt.yticks(X_ticks, fontsize = 7)

plt.grid()

plt.title('Matrix $P^{b}$ - NMC Method', fontsize = 25)
plt.savefig('B_NMC_1203_0.1_1221.eps', format='eps')
#plt.savefig('B_NMC_1203_0.1_1221.png', dpi = 200)
plt.show()
#'''