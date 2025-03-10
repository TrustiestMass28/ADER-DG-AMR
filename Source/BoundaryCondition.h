#ifndef BOUNDARYCONDITION_H
#define BOUNDARYCONDITION_H

#include <iostream>
#include <memory>

#include <AMReX_AmrCore.H>
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

template <typename EquationType>
class ModelEquation;

template <typename NumericalMethodType>
class Solver;

template <typename NumericalMethodType>
class Mesh;

using namespace amrex;


template <typename EquationType>
class BoundaryCondition
{
  public: 
    BoundaryCondition() = default;

    ~BoundaryCondition() = default;

    void operator() (const IntVect& iv, Array4<Real> const& dest,
                    const int dcomp, const int numcomp,
                    GeometryData const& geom, const Real time,
                    const BCRec* bcr, const int bcomp,
                    const int orig_comp) const;

    template <typename... Args>
    void settings(Args... args) {
        _settings(std::forward<Args>(args)...);
    }

    template<typename NumericalMethodType>
    void init(std::shared_ptr<ModelEquation<EquationType>> _model_pde, std::shared_ptr<Solver<NumericalMethodType>> _solver);

    void setModelEquation(std::shared_ptr<ModelEquation<EquationType>> _model_pde);

    void setDirichletBC();

    void setNeumannBC();

    void setBCtype(amrex::Vector<amrex::Vector<int>> _bc_lo_type,
                  amrex::Vector<amrex::Vector<int>> _bc_hi_type);

    void setBCAMREXtype(amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
                        amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi);

    void set_system_curr_component(int _q, int _lev);

    template<typename NumericalMethodType>
    void FillBoundaryCells(std::shared_ptr<Mesh<NumericalMethodType>> mesh,
                          amrex::Vector<amrex::MultiFab>* U_ptr, 
                          int lev, amrex::Real time);

  protected:
    //Ptr used to access numerical method and solver data
    std::shared_ptr<ModelEquation<EquationType>> model_pde;

    //Store amrex BC types in each dim
    amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> bc_lo;
    amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> bc_hi;

    //Stores amrex bc objects identifiers
    //  boundary conditions for U or U_w
    amrex::Vector<amrex::Vector<amrex::BCRec>> bc; 

    //Store type of BC
    //    "dirichlet" == 0
    //    "neumann"   == 1
    //    "periodic"  == 2
    amrex::Vector<amrex::Vector<int>> bc_lo_type;
    amrex::Vector<amrex::Vector<int>> bc_hi_type;

    //Store Periodic/Dirichlet/Neumann BC evalation
    amrex::Vector<amrex::Vector<amrex::Real>> gbc_lo;
    amrex::Vector<amrex::Vector<amrex::Real>> gbc_hi;

    int n_comp;

  private:
      void _settings(amrex::Vector<amrex::Vector<int>> _bc_lo_type,
                    amrex::Vector<amrex::Vector<int>> _bc_hi_type,
                    amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
                    amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi);

      //store the current pde system component (i.e which equation) we are applying bc to
      int curr_q;

      //store current level to which we are applying BCs to
      int curr_lev;
};

template <typename EquationType>    
void BoundaryCondition<EquationType>::setModelEquation(std::shared_ptr<ModelEquation<EquationType>> _model_pde)
{
  model_pde = _model_pde;
}

template <typename EquationType>    
void BoundaryCondition<EquationType>::set_system_curr_component(int _q, int _lev)
{
  curr_q = _q;
  curr_lev = _lev;
}

template <typename EquationType>       
void BoundaryCondition<EquationType>::_settings(amrex::Vector<amrex::Vector<int>> _bc_lo_type,
                                                amrex::Vector<amrex::Vector<int>> _bc_hi_type,
                                                amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
                                                amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi)
{
  setBCtype(_bc_lo_type,_bc_hi_type);
  setBCAMREXtype(_bc_lo,_bc_hi);                                               
}

template <typename EquationType>       
template<typename NumericalMethodType>
void BoundaryCondition<EquationType>::init(std::shared_ptr<ModelEquation<EquationType>> _model_pde,
                                          std::shared_ptr<Solver<NumericalMethodType>> _solver)
{
  setModelEquation(_model_pde);

  //TODO: use solver ptr to pass reference of bc, bc_w
  _solver->init_bc(bc,n_comp);

  gbc_lo.resize(model_pde->Q_model);
  gbc_hi.resize(model_pde->Q_model);

  for(int q=0; q<model_pde->Q_model; ++q){
    gbc_lo[q].resize(AMREX_SPACEDIM);
    gbc_hi[q].resize(AMREX_SPACEDIM); 
  }

  //bc.size(); should be equal to Q
  for(int q=0; q<bc.size(); ++q){
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {  
      int _lo = bc_lo_type[q][d]; 
      int _hi = bc_hi_type[q][d];
      //loop over components
      //low
      if(_lo == 0){//"dirichlet"
        gbc_lo[q][d] =model_pde->pde_BC_gDirichlet(d,-1,q);
      }
      else if(_lo == 1){//"neumann"
        gbc_lo[q][d] =model_pde->pde_BC_gNeumann(d,-1,q);    
      }
      else if(_lo == 2){//"periodic"
        //nothing done
      }

      //high
      if(_hi == 0){//"dirichlet"
        gbc_hi[q][d] =model_pde->pde_BC_gDirichlet(d,1,q);
      }
      else if(_hi == 1){//"neumann"
        gbc_hi[q][d] =model_pde->pde_BC_gNeumann(d,1,q);  
      }
      else if(_hi == 2){//"periodic"
        //nothing done
      }

      for (int n = 0; n < n_comp; ++n){
        bc[q][n].setLo(d,bc_lo[q][d]);
        bc[q][n].setHi(d,bc_hi[q][d]);
      }
    }
  }
  //TODO:maybe use solver to also pass some quadrature or basis info
}

template <typename EquationType>    
void BoundaryCondition<EquationType>::setBCtype(amrex::Vector<amrex::Vector<int>> _bc_lo_type,
                                                amrex::Vector<amrex::Vector<int>> _bc_hi_type)
{
  bc_lo_type.resize(model_pde->Q_model);
  bc_hi_type.resize(model_pde->Q_model);
  
  for(int q=0; q<model_pde->Q_model; ++q){
    bc_lo_type[q].resize(AMREX_SPACEDIM);
    bc_hi_type[q].resize(AMREX_SPACEDIM);
    
    for(int d=0; d<AMREX_SPACEDIM; ++d){
        bc_lo_type[q][d] = _bc_lo_type[q][d];
        bc_hi_type[q][d] = _bc_hi_type[q][d];
    }
  }
}

template <typename EquationType>    
void BoundaryCondition<EquationType>::setBCAMREXtype(amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
                                                    amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi)
{
  bc_lo.resize(model_pde->Q_model);
  bc_hi.resize(model_pde->Q_model);
  
  for(int q=0; q<model_pde->Q_model; ++q){
    for(int d=0; d<AMREX_SPACEDIM; ++d){
      bc_lo[q][d] = _bc_lo[q][d];
      bc_hi[q][d] = _bc_hi[q][d];
    }
  }
}

template <typename EquationType>    
template<typename NumericalMethodType>
void BoundaryCondition<EquationType>::FillBoundaryCells(std::shared_ptr<Mesh<NumericalMethodType>> mesh,
                                                        amrex::Vector<amrex::MultiFab>* U_ptr, 
                                                        int lev, amrex::Real time)
{
  amrex::Geometry geom_l = mesh->get_Geom(lev);
  
  //applies boundary conditions    
  for(int q=0; q<(*U_ptr).size(); ++q){
    //sync MFab internl MFab ghost cells
    //and for external ghost apply Periodic BC if needed/if periodic BC present
    (*U_ptr)[q].FillBoundary(geom_l.periodicity()); 
    
    //if we had all periodic then we pretty much already applied BC and can thus exit
    //if domain isnt all periodic, now apply non-periodic BCs
    if (!(geom_l.isAllPeriodic())){  

      //update in BC object the value of q,lev s.t it knows over which component of datastructures
      //we are applying BCs to
      set_system_curr_component(q,lev);

      amrex::GpuBndryFuncFab<BoundaryCondition> bcf(this);
      amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>> physbcf_bd(geom_l,bc[q],bcf);  
      
      physbcf_bd((*U_ptr)[q], 0, (*U_ptr)[q].nComp(), (*U_ptr)[q].nGrowVect(), time,0);    
      //(MultiFab& mf, int icomp, int ncomp, IntVect const& nghost,real time, int bccomp)
    }
  } 
}
/*
run() //in Simulation
  initialize BC (by Passing NumericalSolver s.t it can access p,...)

  pass bc object to evolve()

//calling BC inside evolve()
  call solver bc method that accept as argument the templated BC object
  this calls bc object methods, pass solver data

//now inside BC object method we have full access to model_pde and can modify the solver passed data


the ::operator() of BC class needs access to modelequaton
if this oeprator cna be modified than chill, just pass model_pde

one solution is to create every time since in eovlve we have model_pde knowledge
so we can instantiate it

otherwise 
*/



  /*
//Boundary Conditions
void FillBoundaryCells(amrex::Vector<amrex::MultiFab>* U_ptr, int lev);

amrex::Real gDirichlet_bc(int d, int side, int q) const;

amrex::Real gNeumann_bc(int d, int side, int q) const;

//Boundary Conditions

FillBoundaryCells

*/
/*
*/
/*
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  SOLVER CONSTRUCTOR

  //from function argument
  


  //AmrDG boundary data




  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  BOUNDARY CONDITION COSNTRUCTOR


*/
/*
*/
    /*

                    
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

#endif 