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

template <typename SolverType,typename ModelEquationType>
class Simulation
{
  public:    
    Simulation(); 
    
    ~Simulation();

    void run();
    
  private:
    int _coord = 0;//cartesian, don't touch

    std::shared_ptr<std::ofstream> ofs;

    std::shared_ptr<ModelEquationType> model;

    std::shared_ptr<SolverType> solver;

};

template <typename SolverType,typename ModelEquationType>
Simulation<SolverType,ModelEquationType>::Simulation() 
{

  solver = std::make_shared<SolverType>();

  model = std::make_shared<ModelEquationType>();
  test();
  //model->test();
  //solver->test();
}

template <typename SolverType,typename ModelEquationType>
Simulation<SolverType,ModelEquationType>::~Simulation() {
  //ofs->close();
}

template <typename SolverType,typename ModelEquationType>
void Simulation<SolverType,ModelEquationType>::run()
{
  //dg_sim->Init();
  //dg_sim->Evolve();
}

#endif // SIMULATION_H

