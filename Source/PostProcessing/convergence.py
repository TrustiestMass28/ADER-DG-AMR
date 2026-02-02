import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def get_convergence(p):
    # Single level     
    #p_E = {1:[0.048870899515791516, 0.01753439920565603,0.0087262139910766756, 0.0057269497617791539],
    #       2:[0.0087790966099986489, 0.0017085238936960308,0.00012726442764960074,1.5981709704366251e-05],
    #       3:[0.0027877884461944696,0.0001746922677643825,7.8290770603552511e-06,3.9752247868729303e-07]}


    # Multi level - Continuous Refinement (10 steps)
    p_E = {
        1: [0.020109813450382732,0.009400131928160858,],
        #2: [0.007403463239853198,0.0010953680374852686, 0.00013840805937993046, 2.2302575656807058e-05,3.4578050603851847e-06]
        2: [0.0017190597072334007,0.00012726442764960074]
    }

    errors = np.array(p_E[p])
    base_N = 8  
    N = np.array([base_N * 2**i for i in range(len(errors))])

    #  Global slope (using all points)
    global_slope, _, _, _, _ = linregress(np.log10(N), np.log10(errors))

    #  Local slopes (between each refinement step)
    # Formula: (log(E2) - log(E1)) / (log(N2) - log(N1))
    log_N = np.log10(N)
    log_E = np.log10(errors)
    local_slopes = (log_E[1:] - log_E[:-1]) / (log_N[1:] - log_N[:-1])

    # --- Print Output ---
    print(f"--- Convergence Analysis for p={p} ---")
    print(f"Theoretical Order (TOC): {p+1}")
    print(f"Global Empirical Order:  {abs(global_slope):.4f}")
    print("-" * 45)
    print(f"{'Step':<6} | {'N':<6} | {'Error':<12} | {'Local Slope':<10}")
    print("-" * 45)
    
    for i in range(len(errors)):
        if i == 0:
            print(f"{i:<6} | {N[i]:<6} | {errors[i]:.4e} | {'-':<10}")
        else:
            # We take the absolute value of the slope to show the order
            print(f"{i:<6} | {N[i]:<6} | {errors[i]:.4e} | {abs(local_slopes[i-1]):.4f}")
    print("-" * 45)

if __name__ == "__main__":
    p = 2
    get_convergence(p)