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


//This BC class implements
//stationary boundary conditions
template <typename EquationType, typename NumericalMethodType>
class BoundaryCondition
{
  public: 
    BoundaryCondition();

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

    void init(std::shared_ptr<ModelEquation<EquationType>> _model_pde, 
              std::shared_ptr<Solver<NumericalMethodType>> _solver,
              std::shared_ptr<Mesh<NumericalMethodType>> _mesh);

    void setModelEquation(std::shared_ptr<ModelEquation<EquationType>> _model_pde);

    void setSolver(std::shared_ptr<Solver<NumericalMethodType>> _solver);

    void setMesh(std::shared_ptr<Mesh<NumericalMethodType>> _mesh);

    void setDirichletBC();

    void setNeumannBC();

    void setBCtype(amrex::Vector<amrex::Vector<int>> _bc_lo_type,
                  amrex::Vector<amrex::Vector<int>> _bc_hi_type);

    void setBCAMREXtype(amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
                        amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi);

    void set_system_curr_component(int _q, int _lev);

    void FillBoundaryCells(amrex::Vector<amrex::MultiFab>* U_ptr, 
                          int lev, amrex::Real time);

  protected:
    //Ptr used to access implemented model equation
    std::weak_ptr<ModelEquation<EquationType>> model_pde;

    //Ptr used to access numerical method and solver data
    std::weak_ptr<Solver<NumericalMethodType>> solver;

    //Ptr used to access mesh/geom data
    std::weak_ptr<Mesh<NumericalMethodType>> mesh;

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

      int Q_model;

      //store the current pde system component (i.e which equation) we are applying bc to
      int curr_q;

      //store current level to which we are applying BCs to
      int curr_lev;

      //: bcf(*this) {}
      std::shared_ptr<amrex::GpuBndryFuncFab<BoundaryCondition<EquationType, NumericalMethodType>>> bcf;
      //amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition>> physbcf_bd;
};

template <typename EquationType, typename NumericalMethodType>   
BoundaryCondition<EquationType,NumericalMethodType>::BoundaryCondition()
{
  bcf = std::make_shared<amrex::GpuBndryFuncFab<BoundaryCondition<EquationType, NumericalMethodType>>>(*this);
}

template <typename EquationType, typename NumericalMethodType>   
void BoundaryCondition<EquationType,NumericalMethodType>::setModelEquation(
                                                          std::shared_ptr<ModelEquation<EquationType>> _model_pde)
{
  model_pde = _model_pde;
}



template <typename EquationType, typename NumericalMethodType>   
void BoundaryCondition<EquationType,NumericalMethodType>::setSolver(std::shared_ptr<Solver<NumericalMethodType>> _solver)
{
  solver = _solver;
}

template <typename EquationType, typename NumericalMethodType>   
void BoundaryCondition<EquationType,NumericalMethodType>::setMesh(std::shared_ptr<Mesh<NumericalMethodType>> _mesh)
{
  mesh = _mesh;
}

template <typename EquationType, typename NumericalMethodType>    
void BoundaryCondition<EquationType,NumericalMethodType>::set_system_curr_component(int _q, int _lev)
{
  curr_q = _q;
  curr_lev = _lev;
}

template <typename EquationType, typename NumericalMethodType>      
void BoundaryCondition<EquationType,NumericalMethodType>::_settings(amrex::Vector<amrex::Vector<int>> _bc_lo_type,
                                                amrex::Vector<amrex::Vector<int>> _bc_hi_type,
                                                amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
                                                amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi)
{
  setBCtype(_bc_lo_type,_bc_hi_type);
  setBCAMREXtype(_bc_lo,_bc_hi);                                               
}

template <typename EquationType, typename NumericalMethodType>
void BoundaryCondition<EquationType,NumericalMethodType>::init(std::shared_ptr<ModelEquation<EquationType>> _model_pde,
                                                              std::shared_ptr<Solver<NumericalMethodType>> _solver,
                                                              std::shared_ptr<Mesh<NumericalMethodType>> _mesh)
{
  setModelEquation(_model_pde);
  
  setSolver(_solver);
  
  setMesh(_mesh);
  

  //TODO: use solver ptr to pass reference of bc, bc_w
  _solver->init_bc(bc,n_comp);

  Q_model = _model_pde->Q_model;
  
  //gbc_lo,gbc_lo accessed inside ModelEquation
  //since we have static BCs, no need to call
  //implemented pde_BC_gDirichlet,pde_BC_gNeumann
  //every time, just do at beginning

  //for each component of the PDE system
  //gbc_lo stores the BC value as defined
  //in the implemented ModelEquation derived
  gbc_lo.resize(Q_model);
  gbc_hi.resize(Q_model);

  for(int q=0; q<Q_model; ++q){
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
        gbc_lo[q][d] =_model_pde->pde_BC_gDirichlet(d,-1,q);
      }
      else if(_lo == 1){//"neumann"
        gbc_lo[q][d] =_model_pde->pde_BC_gNeumann(d,-1,q);    
      }
      else if(_lo == 2){//"periodic"
        //nothing done
      }
      
      //high
      if(_hi == 0){//"dirichlet"
        gbc_hi[q][d] =_model_pde->pde_BC_gDirichlet(d,1,q);
      }
      else if(_hi == 1){//"neumann"
        gbc_hi[q][d] =_model_pde->pde_BC_gNeumann(d,1,q);  
      }
      else if(_hi == 2){//"periodic"
        //nothing done
      }
      
      for (int n = 0; n < n_comp; ++n){
        //for each equation of the PDE system
        //we might have multiple sub components (e.g modes)
        //assign same bc to all of them
        bc[q][n].setLo(d,bc_lo[q][d]);
        bc[q][n].setHi(d,bc_hi[q][d]);
      }
      
    }
  }
}

template <typename EquationType, typename NumericalMethodType>   
void BoundaryCondition<EquationType,NumericalMethodType>::setBCtype(amrex::Vector<amrex::Vector<int>> _bc_lo_type,
                                                amrex::Vector<amrex::Vector<int>> _bc_hi_type)
{
  int Q = _bc_lo_type.size(); //cant get it from model ocz have not apssed ptr yet
  bc_lo_type.resize(Q);
  bc_hi_type.resize(Q);
  
  for(int q=0; q<Q; ++q){
    bc_lo_type[q].resize(AMREX_SPACEDIM);
    bc_hi_type[q].resize(AMREX_SPACEDIM);
    
    for(int d=0; d<AMREX_SPACEDIM; ++d){
        bc_lo_type[q][d] = _bc_lo_type[q][d];
        bc_hi_type[q][d] = _bc_hi_type[q][d];
    }
  }
}

template <typename EquationType, typename NumericalMethodType>
void BoundaryCondition<EquationType,NumericalMethodType>::setBCAMREXtype(amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
                                                    amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi)
{
  int Q = _bc_lo.size();

  bc_lo.resize(Q);
  bc_hi.resize(Q);
  
  for(int q=0; q<Q; ++q){
    for(int d=0; d<AMREX_SPACEDIM; ++d){
      bc_lo[q][d] = _bc_lo[q][d];
      bc_hi[q][d] = _bc_hi[q][d];
    }
  }
}

template <typename EquationType, typename NumericalMethodType>
void BoundaryCondition<EquationType,NumericalMethodType>::FillBoundaryCells(amrex::Vector<amrex::MultiFab>* U_ptr, 
                                                                            int lev, amrex::Real time)
{
  auto _mesh = mesh.lock();

  amrex::Geometry geom_l = _mesh->get_Geom(lev);
  
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

      //TODO:could be improved, maybe pre-cosntuct obejct inside init(). Care about geom object if it gets updated tho
      amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryCondition<EquationType,NumericalMethodType>>> physbcf_bd(geom_l,bc[q],*bcf);  
      
      physbcf_bd((*U_ptr)[q], 0, (*U_ptr)[q].nComp(), (*U_ptr)[q].nGrowVect(), time,0);    
      //(MultiFab& mf, int icomp, int ncomp, IntVect const& nghost,real time, int bccomp)
    }
  } 
}

template <typename EquationType, typename NumericalMethodType>   
void BoundaryCondition<EquationType,NumericalMethodType>::operator() (const IntVect& iv, Array4<Real> const& dest,
                                                  const int dcomp, const int numcomp,
                                                  GeometryData const& geom, const Real time,
                                                  const BCRec* bcr, const int bcomp,
                                                  const int orig_comp) const
{  
  //NB: in case we want to apply different BCs to specific modes, need to pass (in components loop)
  //the comp number to pde_BC_gDirichlet,pde_BC_gNeumann (they need to be modified accordingly)
  //for (int comp = 0; comp < numcomp; ++comp) { ..bc_w[q][comp] and pass comp to functions

  auto _solver = solver.lock();
  auto _model_pde = model_pde.lock();

  const auto lo = geom.Domain().smallEnd();
  const auto hi = geom.Domain().bigEnd();

  //used to hold tmp bc value
  //e.g in case we first evaluate at poitns and then
  //ptorject to modes.
  amrex::Vector<amrex::Real> _bc(_solver->n_pt_bc);

  //The way inner bool checks on bc.lo(dim),bc.hi(dim)
  //work, expects all equation system components to have same BC 
  //(as the first one)
  for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {

    bool bc_found=false;

    if ((bc[curr_q][0].lo(dim) == amrex::BCType::ext_dir) && iv[dim] < lo[dim]){
      //Boundary Low
      for(int m=0; m<_solver->n_pt_bc; ++m){
        if(bc_lo_type[curr_q][dim] == 0)//Dirichlet
        {
          _bc[m] = _model_pde->pde_BC_gDirichlet(curr_q,dim,iv,m,dcomp,numcomp,dest,geom,-1,curr_lev,gbc_lo);
        }
        else if(bc_lo_type[curr_q][dim] == 1)//Neumann
        {
          _bc[m] =  _model_pde->pde_BC_gNeumann(curr_q, dim,iv,m,dcomp,numcomp,dest,geom,-1,curr_lev,gbc_lo);  
        }
      }
      bc_found = true;
    }
    else if ((bc[curr_q][0].hi(dim) == amrex::BCType::ext_dir) && iv[dim] > hi[dim]) {
      //Boundary High
      for(int m=0; m<_solver->n_pt_bc; ++m){
        if(bc_lo_type[curr_q][dim] == 0)//Dirichlet
        {
          _bc[m] = _model_pde->pde_BC_gDirichlet(curr_q,dim,iv,m,dcomp,numcomp,dest,geom,1,curr_lev,gbc_hi);
        }
        else if(bc_lo_type[curr_q][dim] == 1)//Neumann
        {
          _bc[m] =  _model_pde->pde_BC_gNeumann(curr_q, dim,iv,m,dcomp,numcomp,dest,geom,1,curr_lev,gbc_hi);  
        }
      }
      bc_found = true;
    }

    if(bc_found)
    { 
      for (int comp = 0; comp < n_comp; ++comp){ 
        //in case we need to perform other operations, e.g for projection
        //otherwise we simply assign single value of bc to the dest
        dest(iv, dcomp + comp) = _solver->setBC(_bc,comp,dcomp,curr_q, curr_lev);

        //IntVect::TheDimensionVector(dim))[1]
      }           
    }
  }   
}

#endif 