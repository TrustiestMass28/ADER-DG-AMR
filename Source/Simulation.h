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

//class ModelEquation;

//class Solver;
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

  SolverType* solver = new SolverType();
  ModelEquationType* model = new ModelEquationType();

  model->testModel();
  // Set up cross-communication
  //solver->setModelEquation(model);

  //solver->getModelEquation<ModelEquationType>()->testModel();
  //model->setSolver(solver);

  //Enable base classes to access their derived classes
  //  store a pointer to derived numerical method class inside the Solver
  //solver->setNumericalMethod(solver);
  //  store pointer to derived PDE problem class inside ModelEquation
  //model->setModelEquation(model);

  //solver->setModelEquation(model);
  //solver->test();



  //Enable model and numerics part to interact by exchangin pointers 
  //solver->setModelEquation(model);


  //use numerical_pde to call method of NumericalMethod
  //this method is actually virtual and overridden by implementation of AmrDG
  //therefore the AmrDG implementation will be called

  //model_pde->setNumericalMethod(numerical_pde);
  //solver->setModelEquation(model_pde);

  //ofs->open("simulation_output.txt", std::ofstream::out);
  //model_pde->setOfstream(ofs);
  //numerical_pde->setOfstream(ofs);
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

