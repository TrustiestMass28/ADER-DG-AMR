#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "AmrDG.h"
#include "ModelEquation.h"
 
void AmrDG::InitialCondition(int lev)
{
  //Print(*ofs) <<"AmrDG::InitialCondition() "<<lev<<"\n";
  //applies the initial condition to all the solution components modes
  amrex::Vector<amrex::MultiFab *> state_uw(Q);
 
  for(int q=0; q<Q; ++q){
    state_uw[q] = &(U_w[lev][q]); 
  }  
  
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
    amrex::Vector<amrex::FArrayBox *> fab_uw(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > uw(Q);
    
    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_uw[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_uw[0]),true); mfi.isValid(); ++mfi)
    #endif   
    {

      const amrex::Box& bx = mfi.growntilebox();
      //we wil lfill also ghost cells at fine coarse itnerface of fine lvl

      for(int q=0 ; q<Q; ++q){
        fab_uw[q]=&((*(state_uw[q]))[mfi]);
        uw[q]=(*(fab_uw[q])).array();
      } 
      
      for(int q=0; q<Q; ++q){
        amrex::ParallelFor(bx,Np,[&] (int i, int j, int k, int n) noexcept
        { 
          uw[q](i,j,k,n) = Initial_Condition_U_w(lev,q,n,i, j, k);  
        });          
      }
    }   
  }
}

amrex::Real AmrDG::Initial_Condition_U_w(int lev,int q,int n,int i,int j,int k) const
{
  //project initial condition for solution to initial condition for solution modes         
  amrex::Real sum = 0.0;
  for(int m=0; m<qMp_L2proj; ++m)
  {
    sum+= Initial_Condition_U(lev,q,i,j,k,xi_ref_GLquad_L2proj[m])*L2proj_quadmat[n][m];   
  }
  
  return (sum/(RefMat_phiphi(n,n, false, false)));
}

amrex::Real AmrDG::Initial_Condition_U(int lev,int q,int i,int j,int k,
                                        amrex::Vector<amrex::Real> xi) const
{ 
  amrex::Real u_ic;
  u_ic = model_pde->pde_IC(lev,q,i,j,k,xi);

  return u_ic;
}


