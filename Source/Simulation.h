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

#include "Solver.h"
#include "ModelEquation.h"
#include "BoundaryCondition.h"
#include "Mesh.h"

using namespace amrex;

template <typename NumericalMethodType,typename EquationType>
class Simulation
{
  public:    
    Simulation() = default; 
    
    ~Simulation();

    void run();

    template <typename... Args>
    void setNumericalSettings(Args... args);
    
    template <typename... Args>
    void setModelSettings(Args... args);

    template <typename... Args>
    void setBoundaryConditions(Args... args);

    void setGeometrySettings(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, 
                    int _coord, Vector<IntVect> const& _ref_ratios,  
                    Array<int,AMREX_SPACEDIM> const& _is_per, int dtn_regrid = 0, 
                    amrex::Real dt_regrid = 0,int nghost= 1);

    void setIO(int _n_out, amrex::Real _t_out);

    int getQ();

  private:
    std::shared_ptr<std::ofstream> ofs;

    std::shared_ptr<ModelEquation<EquationType>> model;

    std::shared_ptr<Solver<NumericalMethodType>> solver;

    std::shared_ptr<Mesh<NumericalMethodType>> mesh;

    std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>> bdcond;

    //I/O 
    int dtn_outplt;   //data output time-steps interval

    amrex::Real dt_outplt;   //data output physical time interval
};

/*
template <typename NumericalMethodType,typename EquationType>
Simulation<NumericalMethodType,EquationType>::Simulation() 
{}*/

template <typename NumericalMethodType,typename EquationType>
Simulation<NumericalMethodType,EquationType>::~Simulation() {
  //ofs->close();
}

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::run()
{
  mesh->init(solver);
  
  solver->init(model,mesh);

  bdcond->init(model,solver,mesh);
  
  //set initial conditions and average fine->coarse (after Initfromscartch should avg down)
  solver->set_initial_condition(model);
  //TODO:average

  //evolve, pass Model as ptr so we can access its implementation of methods
  solver->evolve(model,bdcond);
}

template <typename NumericalMethodType,typename EquationType>
template <typename... Args>
void Simulation<NumericalMethodType,EquationType>::setNumericalSettings(Args... args) {
  solver = std::make_shared<NumericalMethodType>();
  solver->settings(args...);
}

template <typename NumericalMethodType,typename EquationType>
template <typename... Args>
void Simulation<NumericalMethodType,EquationType>::setModelSettings(Args... args) {
  model = std::make_shared<EquationType>();
  model->settings(args...);
}

template <typename NumericalMethodType,typename EquationType>
template <typename... Args>
void Simulation<NumericalMethodType,EquationType>::setBoundaryConditions(Args... args) {
  bdcond = std::make_shared<BoundaryCondition<EquationType,NumericalMethodType>>();
  bdcond->settings(args...);
}

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::setGeometrySettings(const RealBox& _rb, int _max_level,
                                                                      const Vector<int>& _n_cell, 
                                                                      int _coord, Vector<IntVect> const& _ref_ratios,  
                                                                      Array<int,AMREX_SPACEDIM> const& _is_per,
                                                                      int dtn_regrid, amrex::Real dt_regrid ,int nghost)
{
  mesh = std::make_shared<Mesh<NumericalMethodType>>(_rb,_max_level,_n_cell,_coord,_ref_ratios,_is_per,
                                                      dtn_regrid, dt_regrid, nghost);   
} 

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::setIO(int _n_out, amrex::Real _t_out)
{
  dtn_outplt = _n_out;
  dt_outplt  = _t_out;
}

template <typename NumericalMethodType,typename EquationType>
int Simulation<NumericalMethodType,EquationType>::getQ()
{
  //to be called only after ModelEquation object has been init
  return model->Q_model;
}

#endif // SIMULATION_H

