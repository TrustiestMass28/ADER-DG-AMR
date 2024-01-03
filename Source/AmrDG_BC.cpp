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

#include "AmrDG.h"

void AmrDG::FillBoundaryCells(amrex::Vector<amrex::MultiFab>* U_ptr, int lev)
{
  //applies boundary conditions    
  for(int q=0; q<Q; ++q){
    //sync MFab internl MFab ghost cells
    //and for external ghost apply Periodic BC if needed/if periodic BC present
    (*U_ptr)[q].FillBoundary(geom[lev].periodicity()); 
    
    //if we had all periodic then we pretty much already applied BC and can thus exit
    //if domain isnt all periodic, now apply non-periodic BCs
    if (!(geom[lev].isAllPeriodic())){    
      //apply our own boundary conditions
      BoundaryCondition custom_bc(this, q, lev);
          
      amrex::GpuBndryFuncFab<BoundaryCondition> bcf(custom_bc);
      
      amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>> physbcf_bd(geom[lev],bc_w[q],bcf);  
      
      physbcf_bd((*U_ptr)[q], 0, (*U_ptr)[q].nComp(), (*U_ptr)[q].nGrowVect(), 0.0,0);    
    }
  } 
}
    
AmrDG::BoundaryCondition::BoundaryCondition(AmrDG* _amrdg, int _q, int _lev)
{
  //construct AmrDG class boundary nested class object
  amrdg= _amrdg;
  q = _q;
  lev = _lev; 
  
  for(int d=0; d<AMREX_SPACEDIM; ++d)
  {
    boundary_lo_type[d] = amrdg->bc_lo_type[q][d];
    boundary_hi_type[d] = amrdg->bc_hi_type[q][d];
  }
}

AmrDG::BoundaryCondition::~BoundaryCondition(){}
                    
void AmrDG::BoundaryCondition::operator() (const IntVect& iv, Array4<Real> const& dest,
                                           const int dcomp, const int numcomp,
                                           GeometryData const& geom, const Real time,
                                           const BCRec* bcr, const int bcomp,
                                           const int orig_comp) const
{  
  //NB: in case we want to apply different BCs to specific modes, need to pass (in components loop)
  //the comp number to pde_BC_gDirichlet,pde_BC_gNeumann (they need to be modified accordingly)
  //for (int comp = 0; comp < numcomp; ++comp) { ..bc_w[q][comp] and pass comp to functions
  
  const auto lo = geom.Domain().smallEnd();
  const auto hi = geom.Domain().bigEnd();
  
  amrex::Vector<amrex::Real> bc_val(amrdg->qMp_L2proj);
  
  for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {

    bool bc_found=false;
    
    if(boundary_lo_type[dim] == 0 || boundary_hi_type[dim] == 0)//Dirichlet
    {   
      if ((amrdg->bc_w[q][0].lo(dim) == amrex::BCType::ext_dir) && iv[dim] < lo[dim]){
        for(int m=0; m<amrdg->qMp_L2proj; ++m){
          bc_val[m] = amrdg->sim->model_pde->pde_BC_gDirichlet(q,dim,iv,m,dcomp,numcomp,dest,geom,-1,lev);
        }
        bc_found = true;
      }
      else if ((amrdg->bc_w[q][0].hi(dim) == amrex::BCType::ext_dir) && iv[dim] > hi[dim]) {
        for(int m=0; m<amrdg->qMp_L2proj; ++m){
          bc_val[m] = amrdg->sim->model_pde->pde_BC_gDirichlet(q,dim,iv,m,dcomp,numcomp,dest,geom,1,lev);   
        }          
        bc_found = true;
      }
       
      if(bc_found)
      {
        for (int comp = 0; comp < numcomp; ++comp){ 
          //BC projection u|bc->u_w|bc
          amrex::Real sum = 0.0;
          for(int m=0; m<amrdg->qMp_L2proj; ++m)
          {
            sum+= amrdg->L2proj_quadmat[dcomp +comp][m]*bc_val[m];
          }
          
          sum /=amrdg->RefMat_phiphi(dcomp + comp,dcomp + comp, false, false);
          dest(iv, dcomp + comp) = sum;
        }           
      }           
    }
    else if(boundary_lo_type[dim] == 1 || boundary_hi_type[dim] == 1)//Neumann
    {    
      if ((amrdg->bc_w[q][0].lo(dim) == amrex::BCType::ext_dir) && iv[dim] < lo[dim]){
        for(int m=0; m<amrdg->qMp_L2proj; ++m){
          bc_val[m] = amrdg->sim->model_pde->pde_BC_gNeumann(q, dim,iv,m,dcomp,numcomp,dest,geom,-1,lev);  
        }       
        bc_found = true;
      }      
      else if ((amrdg->bc_w[q][0].hi(dim) == amrex::BCType::ext_dir) && iv[dim] > hi[dim]) {
        for(int m=0; m<amrdg->qMp_L2proj; ++m){
          bc_val[m] = amrdg->sim->model_pde->pde_BC_gNeumann(q, dim,iv,m,dcomp,numcomp,dest,geom,1,lev);   
        }        
        bc_found = true;
      } 
      
      if(bc_found)
      {
        for (int comp = 0; comp < numcomp; ++comp){ 
          //BC projection u|bc->u_w|bc
          amrex::Real sum = 0.0;
          for(int m=0; m<amrdg->qMp_L2proj; ++m)
          {
            sum+= amrdg->L2proj_quadmat[dcomp +comp][m]*bc_val[m];
          }
          
          sum /=amrdg->RefMat_phiphi(dcomp + comp,dcomp + comp, false, false);
          
          dest(iv, dcomp + comp) = sum;     
          
          //IntVect::TheDimensionVector(dim))[1]
        }           
      }
    }
  }
}

amrex::Real AmrDG::gDirichlet_bc(int d, int side, int q) const
{ 
  return sim->model_pde->pde_BC_gDirichlet(d,side,q) ;
}

amrex::Real AmrDG::gNeumann_bc(int d, int side, int q) const
{ 
  return sim->model_pde->pde_BC_gNeumann(d,side,q) ;
}



    /*
    for(int q=0; q<Q; ++q){
      //if we have periodicity in any direction, first fill that
      bool any_periodic = false;
      for (int d = 0; d < AMREX_SPACEDIM; ++d){
        if (geom[lev].isPeriodic(d)) {
          any_periodic = true;
          break;
        }
      }
      if(any_periodic){
       (*U_ptr)[q].FillBoundary(geom[lev].periodicity());     
      }
    }
    */
