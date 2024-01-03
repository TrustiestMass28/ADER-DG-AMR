#ifndef SIMULATION_H
#define SIMULATION_H

#include <string>
#include <limits>
#include <memory>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <AMReX_Interpolater.H>

using namespace amrex;

class ModelEquation;
class AmrDG;

class Simulation
{
  public:    
    Simulation();
    
    ~Simulation();
    
    void settings_case(std::string _equation_type, bool _source_term, 
                        bool _angular_momentum, std::string _test_case);
                          
    void settings_general(const RealBox& _rb, int _max_level,
                          const Vector<int>& _n_cell,
                          Vector<IntVect> const& _ref_ratios, 
                          Array<int,AMREX_SPACEDIM> const& _is_per,
                          amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
                          amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi, 
                          amrex::Vector<amrex::Vector<int>> _bc_lo_type,
                          amrex::Vector<amrex::Vector<int>> _bc_hi_type, 
                          amrex::Real _T,amrex::Real _CFL,int _p, int _t_regrid, 
                          int _t_outplt, std::string _limiter_type, 
                          amrex::Real _TVB_M, 
                          amrex::Vector<amrex::Real> _AMR_TVB_C,
                          amrex::Vector<amrex::Real> _AMR_curl_C, 
                          amrex::Vector<amrex::Real> _AMR_div_C, 
                          amrex::Vector<amrex::Real> _AMR_grad_C, 
                          amrex::Vector<amrex::Real> _AMR_sec_der_C,
                          amrex::Real _AMR_sec_der_indicator, 
                          amrex::Vector<amrex::Real> _AMR_C,
                          int _t_limit);
          
    void run();
    
    ModelEquation* model_pde;
  
    AmrDG* dg_sim;  
    
  private:
    int _coord = 0;//cartesian, don't touch

};

#endif 
