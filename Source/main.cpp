#include <iostream>
#include <AMReX.H>

#include "Simulation.h"
#include "AmrDG.h"
#include "Compressible_Euler.h"

using namespace amrex;

int main(int argc, char* argv[])
{    
    amrex::Initialize(argc,argv);
    {  

      /*-------------------------------------------*/
      //Settings
      std::string test_case = "isentropic_vortex";

      /*-------------------------------------------*/
      //Set-up simulation
      Simulation<AmrDG,Compressible_Euler> sim;

      sim.setModelSettings(test_case);
      sim.run();
            
      /*-------------------------------------------*/
      // wallclock time
      const auto strt_total = amrex::second();
                          
      //sim.run();
                    
      // wallclock time
      auto end_total = amrex::second() - strt_total;
      
      // print wallclock time
      ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
      amrex::Print() << "\nTotal Time: " << end_total << '\n';
    }
    amrex::Finalize();
    
    return 0;
}

/*------------------------------------------------------------------------*/
/* 
--GENERAL CONSIDERATIONS--
TODO: at level L skip covered cells, just then send data fluxes from fine to 
      coarse, if

TODO: function for most commonly used AMR settings, and abiltiy to have presets
TODO: not use global timestepping, but individaul timestepping for each level
TODO: variable naming convention, constructor convention ,...
TODO: limiter class, refinement class
TODO: future, take inspiration from AmrLevel class to extend/modify current code

TODO:-computing ghost values vs exchanging them?
      curently each rhank computes its own ghsot cells values, what about not 
      computing thema dn just exchange them at the end?THis can be done for
      fluxes before Interfacialflux. The important thing are not the internal ones 
      but the boundary ones, we need them them to be updated with the predictor
      We

HPC Considerations:
TODO: AMD PRAGMA SIMD-> compiler directive for vectorization, should be added to 
      innermost loops
TODO: Ordering of varialbes in vector and Mfabs$
TODO: efficient evaluation of Phi(xi) and DPhi(xi) at all nodes uisng iterative method
*/
/*
void settings_numerical(const RealBox& _rb, int _max_level,
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

  dg_sim = new AmrDG(this,_rb,_max_level,_n_cell,_coord, _ref_ratios,_is_per, 
                _bc_lo,_bc_hi,_bc_lo_type,_bc_hi_type,_T,_CFL,_p,
                _t_regrid,_t_outplt,_limiter_type,_TVB_M,
                _AMR_TVB_C, _AMR_curl_C, _AMR_div_C, _AMR_grad_C, _AMR_sec_der_C,
                _AMR_sec_der_indicator, _AMR_C, _t_limit);
*/      


/*------------------------------------------------------------------------*/
/*
      else if(test_case == "double_mach_reflection")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 3.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 1.0;   
        
        n_cell_x = 160;
        n_cell_y = 48;  
        
        CFL = 0.1;
        p  = 1;
        T = 1.0;
        
        t_outplt = 10;
        t_limit = 1;
        t_regrid  = 1;
        
        max_level = 2;
        TVB_M = 1.0;      
      }*/
/*

      else if(test_case == "radial_shock_tube")
      {
        L_x_lo   = -1.0;
        L_x_hi   = 1.0;
      
        L_y_lo   = -1.0;
        L_y_hi   = 1.0; 
        
        n_cell_x = 32;
        n_cell_y = 32;
        T = 0.5;
        CFL = 1.0;
        TVB_M = 1;
        max_level = 2;
        p  = 3;
      }
      else if(test_case == "keplerian_disc")
      {
        L_x_lo   = 0.0;
        L_x_hi   = 6.0;
      
        L_y_lo   = 0.0;
        L_y_hi   = 6.0;   
        
        n_cell_x = 128;
        n_cell_y = 128;
        
        p  = 1;
        CFL = 0.5;
        T = 120.0;
        max_level = 0;
      }
      */
