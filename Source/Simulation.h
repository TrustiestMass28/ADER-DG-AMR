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

using namespace amrex;

template <typename NumericalMethodType,typename EquationType>
class Simulation
{
  public:    
    Simulation(); 
    
    ~Simulation();

    void run();

    template <typename... Args>
    void setNumericalSettings(Args... args) {
      solver->settings(args...);
    }
    
    template <typename... Args>
    void setModelSettings(Args... args) {
          model->settings(args...);
    }

    void setIO(int _n_out, amrex::Real _t_out);

  private:
    int _coord = 0;//cartesian, don't touch

    std::shared_ptr<std::ofstream> ofs;

    std::shared_ptr<ModelEquation<EquationType>> model;

    std::shared_ptr<Solver<NumericalMethodType>> solver;

    //I/O 
    int dtn_outplt;   //data output time-steps interval

    amrex::Real dt_outplt;   //data output physical time interval
};


template <typename NumericalMethodType,typename EquationType>
Simulation<NumericalMethodType,EquationType>::Simulation() 
{
  //Solver base class ptr (construct num method and upcast its ptr to base)
  solver = std::make_shared<NumericalMethodType>();

  model = std::make_shared<EquationType>();
}

template <typename NumericalMethodType,typename EquationType>
Simulation<NumericalMethodType,EquationType>::~Simulation() {
  //ofs->close();
}

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::run()
{
  solver->init(model);
}

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::setIO(int _n_out, amrex::Real _t_out)
{
  dtn_outplt = _n_out;
  dt_outplt  = _t_out;
}

#endif // SIMULATION_H

