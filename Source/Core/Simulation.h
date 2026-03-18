#ifndef SIMULATION_H
#define SIMULATION_H

#include <string>
#include <limits>
#include <memory>
#include <filesystem>  

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
                    Array<int,AMREX_SPACEDIM> const& _is_per, amrex::Vector<amrex::Real> amr_c,
                    int dtn_regrid = 0, amrex::Real dt_regrid = 0,int nghost= 1);

    void setIO(int _n_out, amrex::Real _t_out, std::string _sim_run = "", int _restart_tstep = 0);

    //Set validation mode: if true, use analytical IC at all levels (for convergence tests)
    //If false (default), levels > 0 use projection from coarser level
    void setValidationMode(bool use_analytical_ic);

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

    std::string sim_run;    // optional further subfolder (e.g. "run_1")

    int restart_tstep = 0;
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
  // Build output directory path (all ranks need it for solver->init)
  std::string simulation_data_dir = "../Results/Simulation Data/" + model->model_case;
  if (!sim_run.empty()) {
    simulation_data_dir += "/" + sim_run;
  }

  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::filesystem::create_directories(simulation_data_dir);

    if (restart_tstep == 0) {
      // Fresh run: clean existing plotfiles in this sim's directory
      for (const auto& entry : std::filesystem::directory_iterator(simulation_data_dir)) {
        std::filesystem::remove_all(entry.path());
      }
    }
  }

  //set ptr to solver, other init procedures
  mesh->init(solver);

  //set ptr to mesh and init grids
  solver->init(model,mesh,dtn_outplt,dt_outplt,simulation_data_dir,restart_tstep);

  bdcond->init(model,solver,mesh);
  
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
  model->set_pde_numeric_limits();
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
                                                                      amrex::Vector<amrex::Real> amr_c,
                                                                      int dtn_regrid, amrex::Real dt_regrid ,int nghost)
{
  mesh = std::make_shared<Mesh<NumericalMethodType>>(_rb,_max_level,_n_cell,_coord,_ref_ratios,_is_per,amr_c,
                                                      dtn_regrid, dt_regrid, nghost);  
} 

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::setIO(int _n_out, amrex::Real _t_out, std::string _sim_run, int _restart_tstep)
{
  dtn_outplt    = _n_out;
  dt_outplt     = _t_out;
  sim_run       = _sim_run;
  restart_tstep = _restart_tstep;
}

template <typename NumericalMethodType,typename EquationType>
int Simulation<NumericalMethodType,EquationType>::getQ()
{
  //to be called only after ModelEquation object has been init
  return model->Q_model;
}

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::setValidationMode(bool use_analytical_ic)
{
  solver->setValidationMode(use_analytical_ic);
}

#endif // SIMULATION_H

