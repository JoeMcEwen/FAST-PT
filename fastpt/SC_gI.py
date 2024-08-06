from __future__ import division
import numpy as np
from .J_table import J_table 
import sys
from time import time
from numpy import log, sqrt, exp, pi
from scipy.signal import fftconvolve as convolve



# Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
def P_SC_gI2(k,P):

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL
	cut=3
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

	

	Z1=lambda r : (((4.*r*(45.-165.*r**2+379.*r**4+45.*r**6)+45.*(-1.+r**2)**4*log((-1.+r)**2)-90.*(-1.+r**2)**4*log(np.absolute(1.+r)))/(2016.*r**3))-68./63*r**2)/2.
	Z1_high=lambda r : ((-16*r**2)/21. + (16*r**4)/49. - (16*r**6)/441. - (16*r**8)/4851. - 16*r**10/21021.- 16*r**12/63063.)/2.
	Z1_low=lambda r: ((-16*r**0)/21.+ 16/(49.*r**2) - 16/(441.*r**4) - 16/(4851.*r**6) - 16/(21021.*r**8) - 16/(63063.*r**10)- 16/(153153.*r**12)  )/2.
    

	f_mid_low=Z1(exp(-mid_low_s))*exp(-mid_low_s)
	f_mid_high=Z1(exp(-mid_high_s))*exp(-mid_high_s)
	f_high = Z1_high(exp(-high_s))*exp(-high_s)
	f_low = Z1_low(exp(-low_s))*exp(-low_s)

	f	=np.hstack((f_low,f_mid_low,-0.2381002916036672,f_mid_high,f_high))
	

	g= convolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	SC_gI2= k**3/(2.*pi**2) * P*g_k
	return SC_gI2


def SC_gI1():
	# Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
	sc_mat_1= np.array([[0,0,0,2,0,1./1]], dtype=float)
	table=np.zeros(10,dtype=float)
	for i in range(sc_mat_1.shape[0]):
		x1=J_table(sc_mat_1[i])
		table=np.row_stack((table,x1))
	return table[1:,:]
def SC_gI3():
	# Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
	sc_mat_3= np.array([[0,0,0,0,4,16./245],\
          [1,-1,0,0,3,1./5],\
          [-1,1,0,0,3,1./5],\
          [1,-1,0,0,1,2./15],\
          [-1,1,0,0,1,2./15],\
		  [0,0,0,0,2,254./441],\
		  [0,0,0,0,0,8./315]], dtype=float)
	table=np.zeros(10,dtype=float)
	for i in range(sc_mat_3.shape[0]):
		x3=J_table(sc_mat_3[i])
		table=np.row_stack((table,x3))
	return table[1:,:]
def SC_gI4():
	# Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
	sc_mat_4= np.array([[0,0,0,0,0,-2./9],\
          [0,0,0,2,0,2./9],\
          [0,0,0,2,2,2./3]], dtype=float)
	table=np.zeros(10,dtype=float)
	table=np.zeros(10,dtype=float)
	for i in range(sc_mat_4.shape[0]):
		x4=J_table(sc_mat_4[i])
		table=np.row_stack((table,x4))
	return table[1:,:]
def SC_gI5():
	# Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
	sc_mat_5= np.array([[0,0,0,0,0,-1./6],\
          [0,0,0,2,0,-1./3],\
          [0,0,2,0,0,-1./3],\
		  [0,0,0,0,2,-1./3],\
		  [0,0,1,1,1,3./2]], dtype=float)
	table=np.zeros(10,dtype=float)
	table=np.zeros(10,dtype=float)
	for i in range(sc_mat_5.shape[0]):
		x5=J_table(sc_mat_5[i])
		table=np.row_stack((table,x5))
	return table[1:,:]
def SC_gI6():
	# Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
	sc_mat_6= np.array([[0,0,0,0,0,-2./45],\
          [0,0,1,1,1,2./5],\
          [0,0,0,0,2,-11./63],\
          [0,0,0,2,2,-2./9],\
          [0,0,2,0,2,-2./9],\
		  [0,0,1,1,3,3./5],\
		  [0,0,0,0,4,-4./35]], dtype=float)
	table=np.zeros(10,dtype=float)
	for i in range(sc_mat_6.shape[0]):
		x6=J_table(sc_mat_6[i])
		table=np.row_stack((table,x6))
	return table[1:,:]









	