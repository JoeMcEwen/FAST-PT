"""An example script to run FASTPT
Initializes and calculates all quantities supported by FASTPT
Makes a plot for P_22 + P_13
"""
from time import time

import numpy as np
import matplotlib.pyplot as plt

import fastpt as fpt
from fastpt import FASTPT

#Version check
print('This is FAST-PT version', fpt.__version__)

# load the data file

d=np.loadtxt('Pk_test.dat')
# declare k and the power spectrum
k=d[:,0]; P=d[:,1]

# set the parameters for the power spectrum window and
# Fourier coefficient window
#P_window=np.array([.2,.2])
C_window=.75
#document this better in the user manual

# padding length
n_pad=int(0.5*len(k))
to_do=['all']

# initialize the FASTPT class
# including extrapolation to higher and lower k
# time the operation
t1=time()
fpt=FASTPT(k,to_do=to_do,low_extrap=-5,high_extrap=3,n_pad=n_pad)
t2=time()

# calculate 1loop SPT (and time the operation)
P_spt=fpt.one_loop_dd(P,C_window=C_window)

t3=time()
print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')

#calculate tidal torque EE and BB P(k)
P_IA_tt=fpt.IA_tt(P,C_window=C_window)
P_IA_ta=fpt.IA_ta(P,C_window=C_window)
P_IA_mix=fpt.IA_mix(P,C_window=C_window)
P_RSD=fpt.RSD_components(P,1.0,C_window=C_window)
P_kPol=fpt.kPol(P,C_window=C_window)
P_OV=fpt.OV(P,C_window=C_window)
sig4=fpt.sig4

# make a plot of 1loop SPT results

ax=plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$P(k)$', size=30)
ax.set_xlabel(r'$k$', size=30)

ax.plot(k,P,label='linear')
ax.plot(k,P_spt[0], label=r'$P_{22}(k) + P_{13}(k)$' )
#ax.plot(k,P_IA_mix[0])
#ax.plot(k,-1*P_IA_mix[0],'--')
#ax.plot(k,P_IA_mix[1])
#ax.plot(k,-1*P_IA_mix[1],'--')

plt.legend(loc=3)
plt.grid()
plt.show()

P_IA_tidal=fpt.IA_tidal(P,C_window=C_window)
ax=plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$P(k)$', size=30)
ax.set_xlabel(r'$k$', size=30)

ax.plot(k,abs(P_IA_tidal[0]),label='0')
ax.plot(k,abs(P_IA_tidal[1]),label='1')
ax.plot(k,abs(P_IA_tidal[2]),label='2')
ax.plot(k,abs(P_IA_tidal[3]),label='3')
ax.plot(k,abs(P_IA_tidal[4]),label='4')

plt.legend(loc=3)
plt.grid()
plt.show()
