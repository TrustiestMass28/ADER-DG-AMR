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
using namespace amrex;

#include "Solver.h"
#include "ModelEquation.h"

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

  private:
    int _coord = 0;//cartesian, don't touch

    std::shared_ptr<std::ofstream> ofs;

    std::shared_ptr<ModelEquation<EquationType>> model;

    std::shared_ptr<Solver<NumericalMethodType>> solver;
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

#endif // SIMULATION_H

