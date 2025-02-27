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
#include "ModelEquation.h"

//BOUNDARY CODNITION

              





amrex::Vector<amrex::Vector<amrex::Real>> gDbc_lo;
amrex::Vector<amrex::Vector<amrex::Real>> gDbc_hi;

amrex::Vector<amrex::Vector<amrex::Real>> gNbc_lo;
amrex::Vector<amrex::Vector<amrex::Real>> gNbc_hi;


  
//Boundary Conditions
void FillBoundaryCells(amrex::Vector<amrex::MultiFab>* U_ptr, int lev);

amrex::Real gDirichlet_bc(int d, int side, int q) const;

amrex::Real gNeumann_bc(int d, int side, int q) const;

//Boundary Conditions
amrex::Vector<amrex::Vector<amrex::BCRec>> bc_w; 
amrex::Vector<amrex::Vector<int>> bc_lo_type;
amrex::Vector<amrex::Vector<int>> bc_hi_type;
 
class BoundaryCondition
{
  public: 
    BoundaryCondition();

    virtual ~BoundaryCondition();

    void init(int _q, int _lev);
    
    void operator() (const IntVect& iv, Array4<Real> const& dest,
                    const int dcomp, const int numcomp,
                    GeometryData const& geom, const Real time,
                    const BCRec* bcr, const int bcomp,
                    const int orig_comp) const;

    void setNumericalMethod(NumericalMethodType* _numme);

  protected:
    //Ptr used to access numerical method and solver data
    NumericalMethodType* numme;

    //Store type of BC in each dim
    int boundary_lo_type[AMREX_SPACEDIM];
    int boundary_hi_type[AMREX_SPACEDIM];¨

    //int q;
    //int lev;
};

/*
FillBoundaryCells


*/

//SET THE BC
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


/*

  amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo, 
  amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi, 
  amrex::Vector<amrex::Vector<int>> _bc_lo_type, 
  amrex::Vector<amrex::Vector<int>> _bc_hi_type,,


  //Amrex boundary data
  bc_w.resize(Q,amrex::Vector<amrex::BCRec>(Np));
  for(int q=0; q<Q; ++q){
    for (int n = 0; n < Np; ++n){
      for (int d = 0; d < AMREX_SPACEDIM; ++d)
      {   
        bc_w[q][n].setLo(d,_bc_lo[q][d]);
        bc_w[q][n].setHi(d,_bc_hi[q][d]);
      }
    }
  }

  //AmrDG boundary data
  bc_lo_type.resize(Q);
  bc_hi_type.resize(Q);

  gDbc_lo.resize(Q);
  gDbc_hi.resize(Q);

  gNbc_lo.resize(Q);
  gNbc_hi.resize(Q);

  for(int q=0; q<Q; ++q){
    bc_lo_type[q].resize(AMREX_SPACEDIM);
    bc_hi_type[q].resize(AMREX_SPACEDIM);
    
    gDbc_lo[q].resize(AMREX_SPACEDIM);
    gDbc_hi[q].resize(AMREX_SPACEDIM); 
    
    gNbc_lo[q].resize(AMREX_SPACEDIM);
    gNbc_hi[q].resize(AMREX_SPACEDIM);  
    
    for(int d=0; d<AMREX_SPACEDIM; ++d){
        bc_lo_type[q][d] = _bc_lo_type[q][d];
        bc_hi_type[q][d] = _bc_hi_type[q][d];
        
        gDbc_lo[q][d] =gDirichlet_bc(d,-1,q);
        gDbc_hi[q][d] =gDirichlet_bc(d,1,q);
        
        gNbc_lo[q][d] =gNeumann_bc(d,-1,q);
        gNbc_hi[q][d] =gNeumann_bc(d,1,q);             
    }
  }

*/


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
    /*
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
          //bc_val[m] = amrdg->sim->model_pde->pde_BC_gDirichlet(q,dim,iv,m,dcomp,numcomp,dest,geom,-1,lev);
        }
        bc_found = true;
      }
      else if ((amrdg->bc_w[q][0].hi(dim) == amrex::BCType::ext_dir) && iv[dim] > hi[dim]) {
        for(int m=0; m<amrdg->qMp_L2proj; ++m){
          //bc_val[m] = amrdg->sim->model_pde->pde_BC_gDirichlet(q,dim,iv,m,dcomp,numcomp,dest,geom,1,lev);   
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
          //bc_val[m] = amrdg->sim->model_pde->pde_BC_gNeumann(q, dim,iv,m,dcomp,numcomp,dest,geom,-1,lev);  
        }       
        bc_found = true;
      }      
      else if ((amrdg->bc_w[q][0].hi(dim) == amrex::BCType::ext_dir) && iv[dim] > hi[dim]) {
        for(int m=0; m<amrdg->qMp_L2proj; ++m){
          //bc_val[m] = amrdg->sim->model_pde->pde_BC_gNeumann(q, dim,iv,m,dcomp,numcomp,dest,geom,1,lev);   
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
  return model_pde->pde_BC_gDirichlet(d,side,q) ;
}

amrex::Real AmrDG::gNeumann_bc(int d, int side, int q) const
{ 
  return model_pde->pde_BC_gNeumann(d,side,q) ;
}
*/


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
