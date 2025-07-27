import pathlib
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy as sp

AMREX_SPACEDIM = 2
p = 2
Np = None
       
def dof_mapper():
    ctr = 0
    mat_idx_s = np.ndarray(shape=(Np,AMREX_SPACEDIM))
    if AMREX_SPACEDIM==2:
        for i in range(p+1):
            for j in range((p-i)+1):
                mat_idx_s[ctr][0] = i;
                mat_idx_s[ctr][1] = j;
                ctr +=1
    elif AMREX_SPACEDIM ==1:
        for i in range(p+1):
            mat_idx_s[ctr][0] = i;
            ctr +=1
    return mat_idx_s

def get_num_modes():
    global Np
    Np = int(np.math.factorial(p+AMREX_SPACEDIM)/(np.math.factorial(p)*np.math.factorial(AMREX_SPACEDIM)))
   
def Phi(idx, x, base_dof_map):
    res = 1.0
    for d in range(AMREX_SPACEDIM):
        nd =base_dof_map[idx][d]       
        coeff = sp.special.legendre(nd)
        value = np.polyval(coeff, x[d])
        
        res*=value

    return res
    
def get_U_from_U_w(u_w,xi,base_dof_map):
    u = 0.0;
    for i in range(Np):
        u+=u_w[i]*Phi(i, xi, base_dof_map)
    return u;

def main():
    get_num_modes()
    base_idx = dof_mapper()

    N = 100
    x_points = np.linspace(-1, 1, N)
    xm_points = np.linspace(-1, 1, N)
    xp_points = np.linspace(-1, 1, N)
    
    y_points = np.ones(N)*-0.5
    c_points = np.column_stack((x_points, y_points))
    fm_points =np.column_stack((xm_points, 0.5*y_points))
    fp_points =np.column_stack((xp_points, 0.5*y_points))
    #"""""
    
    uw_c_pre = [8.572916667,0.283203125,0.003255208333,0.169921875,-9.992007222e-16,0.001953125]
    #uw_c_post = [1.610110046,0.01473788146]
    #uw_fmm = [1.551971071,0.02983353782,0.02983353782]
    #uw_fpm = [1.611638147,0.02983353782,0.02983353782]
    uw_fmm = [8.346354167,0.1391601562,0.0008138020833,0.08349609375, -1.998401444e-15,0.00048828125]
    uw_fpm = [8.516276042,0.1391601562,0.0008138020833,0.08642578125,-9.992007222e-16,0.00048828125]
    uw_favg = (np.array(uw_fmm)+np.array(uw_fpm))/2.0
    print(uw_favg)
    #"""""
    #uw_c_pre = [1.611638147,4.081323655,4.081323655]
    #uw_fmm = [1.553503918,3.859950665,3.859950665]
    #uw_fpm = [1.610187433,4.000821329,4.151683367]
    
    #uw_c_pre = [1.604511138,0.06038403536,0.06038403536]
    #uw_fmm = [1.60151611,0.0598296372,0.0598296372]
    #uw_fpm = [1.60151611,0.05408777964,0.06607795781]
    
    """""
    uw_c_pre = [1.777503144,0.07766404015]
    uw_c_post = [1.776531493,0.09639467239]
    uw_fm = [1.737995154,0.08107050145]
    uw_fp = [1.815067831,0.07329015556]
    """""
    
    u_c_pre = []
    u_c_post = []
    u_fmm = []
    u_fpm = []
    u_fpm = []
    u_fmp = []
    u_fpp = []
    u_fm = []
    u_fp = []
    u_favg=[]
    for pt in range(len(c_points)):
        u_c_pre.append(get_U_from_U_w(uw_c_pre,c_points[pt],base_idx))
        u_favg.append(get_U_from_U_w(uw_favg,c_points[pt],base_idx))
        #u_c_post.append(get_U_from_U_w(uw_c_post,c_points[pt],base_idx))
        #u_c_post.append(get_U_from_U_w(uw_c_post,c_points[pt],base_idx))
       
    for pt in range(len(xm_points)):
        u_fmm.append(get_U_from_U_w(uw_fmm,fm_points[pt],base_idx))
        u_fpm.append(get_U_from_U_w(uw_fpm,fp_points[pt],base_idx))
        #u_fm.append(get_U_from_U_w(uw_fm,fm_points[pt],base_idx))
        #u_fp.append(get_U_from_U_w(uw_fp,fp_points[pt],base_idx))
           
    xm_points_tmp=0.5*xm_points-0.5;
    xp_points_tmp=0.5*xp_points+0.5;
    plt.plot(x_points, u_c_pre, label='coarse Uh(x,t)-pre avgdown', color='black')
    #plt.plot(x_points, u_favg, label='fine avg', color='red')
    plt.axvline(0, color='black', linewidth=2.0, linestyle='--') 
    #plt.plot(x_points, u_c_post, label='coarse Uh(x,t)-post avgdown', color='red')
    plt.plot(xm_points_tmp, u_fmm, label='fine Uh(x,t) [-1,0]', color='green')
    plt.plot(xp_points_tmp, u_fpm, label=' fine Uh(x,t) [0,1]', color='green')
    plt.xlabel('x')
    plt.title("Coarse cell (i,j)=(13,13)")
    plt.ylabel('U(x,t)')
    plt.legend()
    plt.grid()
    plt.savefig("AMR_Modes_default_avgdown2d.png")



if __name__ == "__main__":
    main()


