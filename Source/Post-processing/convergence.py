import yt
import pathlib
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def main_convergence():
    
    #Volume/domain size [-1,1]x[-1,1]x[-1,1]
    D = 2
    v = 2**D
    #V_ary   = np.square(np.array([v,v]))
    N   =  np.array([16,32,64,128])
    #h   = np.power(np.divide(V_ary,N), np.array([1.0/D,1.0/D]))
    p = 1
    L = 2
    
    #p_E=[p0_E,p1_E,p2_E,p3_E] pk_E=[E_N_0,E_N_1,...]
    flag_plot = False
    if L==1:       
        p_E = []
    
    elif L==2:       
        p_E = {0: [], 1:[0.012723042539771219,0.0037520311879360949,0.0025053903332453477, 0.0013936300901864704]}
        
    for i in range(len(N)-1):
        alpha = abs((np.log10(p_E[p][i])-np.log10(p_E[p][i+1]))/(np.log10(N[i])-np.log10(N[i+1])))
    slope, intercept, r_value, p_value, std_err = linregress(np.log10(N), np.log10(p_E[p]))
    print("EOC p"+str(p)+": "+str(abs(slope)))  
    print("TOC p"+str(p)+": "+str(p+1))  



    """""
    if flag_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(N, p_E[p], 'o-', label='$DG-p=$'+str(p))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$N$ [-]', fontsize=20)
        if L==1:
            plt.ylabel('$\epsilon_{L^1(\Omega)}(\\rho)$ [-]', fontsize=20)
        elif L==2:
            plt.ylabel('$\epsilon_{L^2(\Omega)}(\\rho)$ [-]', fontsize=20)
        #plt.title('ADER-D')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig("plots/2d_advectionx_grid_convergence.png")
    """""
if __name__ == "__main__":
    main_convergence()

