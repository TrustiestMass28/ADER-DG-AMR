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
      /*------------------SETTINGS-----------------*/

      // NUMERICAL
      constexpr int p = 2;           // polynomial degree
      std::string simulation_case    = "isentropic_vortex";

      amrex::Real T                  = 10.0;
      amrex::Real c_dt               = 0.5;  // CFL safety factor

      // LIMITER (set type="" or t_limit<=0 to disable)
      std::string limiter_type       = "";
      amrex::Real TVB_M              = 0.0;
      int t_limit                    = -1;

      // AMR  (max_level=0: single level, max_level>0: multi-level)
      int max_level                  = 0;

      // VALIDATION MODE
      // true:  use analytical IC at all levels (convergence tests)
      // false: levels > 0 use projection from coarser level (normal AMR)
      bool validation_mode           = true;

      // I/O
      int restart_tstep              = 0;   // set >0 to restart from plotfile
      int dtn_outplt                 = -1;  // output every n timesteps  (<0 = off)
      amrex::Real dt_outplt          = -1;  // output every dt time      (<0 = off)
      std::string sim_run            = "";  // optional subfolder, e.g. "run_1"
      int dtn_regrid                 = -1;  // regrid every n timesteps  (<0 = off)
      amrex::Real dt_regrid          = 10;  // regrid every dt time      (<0 = off)
      int nghost                     = 1;   // ghost cells — do not change

      // GEOMETRY
      int n_cell_x, n_cell_y, n_cell_z;
      amrex::Real L_x_lo, L_x_hi;
      amrex::Real L_y_lo, L_y_hi;
      amrex::Real L_z_lo, L_z_hi;
      int coord                      = 0;

      // BOUNDARY CONDITIONS (per dimension — applied uniformly to all solution components)
      amrex::Array<int,AMREX_SPACEDIM> is_periodic;
      amrex::Array<int,AMREX_SPACEDIM> bc_lo_dim, bc_hi_dim;         // AMReX BC type
      amrex::Array<int,AMREX_SPACEDIM> bc_lo_type_dim, bc_hi_type_dim; // 0=Dirichlet, 1=Neumann, 2=periodic

      /*-------------------------------------------*/
      /*----------CASE-SPECIFIC OVERRIDES----------*/

      if(simulation_case == "isentropic_vortex")
      {
            T             = 10.0;
            c_dt          = 0.9;
            max_level     = 1;
            limiter_type  = "";
            dtn_regrid    = 1;
            dt_regrid     = -1;

            L_x_lo = 0.0;  L_x_hi = 10.0;  n_cell_x = 16;
            L_y_lo = 0.0;  L_y_hi = 10.0;  n_cell_y = 16;
            L_z_lo = 0.0;  L_z_hi = 0.0;   n_cell_z = 1;

            is_periodic[0]     = 1;
            is_periodic[1]     = 1;
            for(int d=0; d<AMREX_SPACEDIM; ++d){
                  bc_lo_dim[d]      = BCType::int_dir;
                  bc_hi_dim[d]      = BCType::int_dir;
                  bc_lo_type_dim[d] = 2;
                  bc_hi_type_dim[d] = 2;
            }
      }
      else if(simulation_case == "kelvin_helmolz_instability")
      {
            T             = 7.0;
            c_dt          = 0.5;
            max_level     = 2;
            limiter_type  = "TVB";
            TVB_M         = 5000.0;
            t_limit       = 1;
            dt_outplt     = 0.05;
            dtn_regrid    = -1;
            dt_regrid     = 0.05;
            restart_tstep = 110708;
            //NB: large M*h² tolerates large jumps (good for vortices, bad near shocks)
            //    small M*h² is strict (clean shocks, smooth features clipped)

            L_x_lo = 0.0;  L_x_hi = 1.0;  n_cell_x = 64;
            L_y_lo = 0.0;  L_y_hi = 1.0;  n_cell_y = 64;
            L_z_lo = 0.0;  L_z_hi = 0.0;  n_cell_z = 1;

            is_periodic[0]     = 1;
            is_periodic[1]     = 1;
            for(int d=0; d<AMREX_SPACEDIM; ++d){
                  bc_lo_dim[d]      = BCType::int_dir;
                  bc_hi_dim[d]      = BCType::int_dir;
                  bc_lo_type_dim[d] = 2;
                  bc_hi_type_dim[d] = 2;
            }
      }

      // AMR tagging thresholds (allocated after max_level is set)
      amrex::Vector<amrex::Real> amr_c(max_level, 1.0);
      amrex::Vector<amrex::Real> amr_tvb_c(max_level + 1, 1.0);

      if(simulation_case == "isentropic_vortex")
      {
            if(max_level > 0) amr_c[0] = 1.4;
      }
      else if(simulation_case == "kelvin_helmolz_instability")
      {
            //t=0->86906:    amr_c = {1500, 2500}
            //t=86906->106168: amr_c = {1100, 2100}
            //t=106168+:
            amr_c[0] = 1300.0;
            amr_c[1] = 2100.0;
      }

      // Build geometry objects
      amrex::Vector<int> n_cell{AMREX_D_DECL(n_cell_x, n_cell_y, n_cell_z)};
      amrex::RealBox domain{{AMREX_D_DECL(L_x_lo, L_y_lo, L_z_lo)},
                            {AMREX_D_DECL(L_x_hi, L_y_hi, L_z_hi)}};

      // Refinement ratio — keep at 2; some internals assume it
      amrex::Vector<amrex::IntVect> amr_ratio;
      for(int l = 0; l < max_level; ++l)
            amr_ratio.push_back(amrex::IntVect(AMREX_D_DECL(2, 2, 2)));

      /*-------------------------------------------*/
      /*-----------SIMULATION CONSTRUCTION---------*/

      Simulation<AmrDG<p>,Compressible_Euler> sim;

      sim.setModelSettings(simulation_case);

      sim.setNumericalSettings(T, c_dt, limiter_type, TVB_M, amr_tvb_c, t_limit);
      sim.setValidationMode(validation_mode);
      sim.setIO(dtn_outplt, dt_outplt, sim_run, restart_tstep);
      sim.setBoundaryConditions(bc_lo_type_dim, bc_hi_type_dim, bc_lo_dim, bc_hi_dim);
      sim.setGeometrySettings(domain, max_level, n_cell, coord, amr_ratio,
                              is_periodic, amr_c, dtn_regrid, dt_regrid, nghost);

      /*-------------------------------------------*/
      /*--------------------RUN--------------------*/

      const auto strt_total = amrex::second();
      sim.run();
      auto end_total = amrex::second() - strt_total;
      ParallelDescriptor::ReduceRealMax(end_total, ParallelDescriptor::IOProcessorNumber());
      //amrex::Print() << "\nTotal Time: " << end_total << '\n';

    }
    amrex::Finalize();

    return 0;
}

      /*-------------------------------------------*/
      /*--------------------RUN--------------------*/


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
