#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>
#include <cmath>
#include <math.h>
#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include "Simulation.h"
#include "Compressible_Euler.h"
#include "Advection.h"
#include "AmrDG.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

using namespace amrex;

Simulation::Simulation() : model_pde(nullptr), dg_sim(nullptr){}

Simulation::~Simulation() {
  delete dg_sim;
  delete model_pde;
}

void Simulation::settings_case(std::string _equation_type, bool _source_term, 
                            bool _angular_momentum, std::string _test_case)
{
  if(_equation_type == "Compressible_Euler"){
    model_pde = new Compressible_Euler(this,_test_case,_equation_type,
                                      _angular_momentum,_source_term);
  }
  else if(_equation_type == "Advection"){
    model_pde = new Advection(this,_test_case,_equation_type,
                                      _angular_momentum,_source_term);
  }
}

void Simulation::settings_general(const RealBox& _rb, int _max_level,
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
                                  int _t_limit)
{
  dg_sim = new AmrDG(this,_rb,_max_level,_n_cell,_coord, _ref_ratios,_is_per, 
                _bc_lo,_bc_hi,_bc_lo_type,_bc_hi_type,_T,_CFL,_p,
                _t_regrid,_t_outplt,_limiter_type,_TVB_M,
                _AMR_TVB_C, _AMR_curl_C, _AMR_div_C, _AMR_grad_C, _AMR_sec_der_C,
                _AMR_sec_der_indicator, _AMR_C, _t_limit);
}    

void Simulation::run()
{
  dg_sim->Init();
  dg_sim->Evolve();
}


