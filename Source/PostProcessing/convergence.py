import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def get_convergence(p):
    # Single level     
    #p_E = {1:[0.048870899515791516, 0.01753439920565603,0.0087262139910766756, 0.0057269497617791539],
    #       2:[0.0087790966099986489, 0.0017085238936960308,0.00012726442764960074,1.5981709704366251e-05],
    #       3:[0.0027877884461944696,0.0001746922677643825,7.8290770603552511e-06,3.9752247868729303e-07]}

    #t=0 Analytical IC vs Coarse->Fine projection IC and correctness of L2 norm computation
    #   IC Analytical
    p_E = {2:[0.00042583108055586113,5.5360950076592396e-05,7.0072370781971762e-06]}
    #   IC projection
    p_E = {2:[0.0030529774696598099 ,0.00042516378132576201,5.5019955705014079e-05]}

    #Note: Lose approximately 1 refinement level of accuracy (error-wise) by interpolating
    #but both mantain same order of convergence (slope-wise).
    # The fine cells inherit the coarse polynomial's
    #accuracy, which is O(h_coarse^{p+1}) = O((2h_fine)^{p+1}) = 2^{p+1} * O(h_fine^{p+1}).

    # Multi level - static refinement
    #   Analytical IC
    #p_E = {2:[0.0065278464577034796,0.0011023970814282044, 6.959679923363609e-05, 7.6752757712181714e-06]}

    #   Projected IC 
    #p_E = {2:[0.0065161678304828134,0.0011138811390704361,6.9862275494816416e-05]}

    # Multi level - dynamic refinement (dt_regrid = 2.0; approx 5 regrids) amr_c[l] = 1.4
    #p_E = {2:[0.0033186281845524477,0.00042112800767480382,6.4383288440806528e-05 ]}

    # Multi level - dynamic refinement (dt_regrid = 0.5; approx 20 regrids) amr_c[l] = 1.4
    #p_E = {2:[0.0017060809914467333 ,0.00016645002888053664,3.0195089070121167e-05]}

    # Multi level - dynamic refinement (every 1step) amr_c[l] = 1.4
    #p_E = {2:[0.0018782008100473721,0.00018983031022766459,2.8531662628985067e-05]}
                           
    # Multi level - dynamic refinement (every 1step) amr_c[l] = 1.4, NO REFLUX
    #p_E = {2:[0.0022590299144886073,0.00043389295945724159,9.9539998686174079e-05]}


    #Conclusion
    #  - Single level: clean p+1 convergence for p=1,2,3                                           
    #    - t=0 diagnostics: L2 norm correct, interpolation correct at order p+1                    
    #    - Multi-level static (analytical IC): clean 3rd order with 4 data points                    
    #    - Multi-level static (projected IC): same as analytical — confirms IC projection doesn't    
    #    affect final error
    #    - Dynamic regridding: accumulation tradeoff between error reduction 
    #    from more frequent regrids and error increase from interpolation.




    
    
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