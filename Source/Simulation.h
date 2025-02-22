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

template <typename NumericalMethodType,typename ModelEquationType>
class Simulation
{
  public:    
    Simulation(); 
    
    ~Simulation();

    void run();

    template <typename... Args>
    void setNumericalSettings(Args... args) {
          num_method->settings(args...);
    }
    
  private:
    int _coord = 0;//cartesian, don't touch

    std::shared_ptr<std::ofstream> ofs;

    std::shared_ptr<ModelEquationType> model;

    std::shared_ptr<NumericalMethodType> num_method;

    std::shared_ptr<Solver<NumericalMethodType>> solver; 

};



//Solver object
//Numerical base object
//  AmrDG object
//

//pass AmrDG
//init AmrDG
//init templated derived NumericalMethod<AmrDG>  : public Solver
//inti base Solver


//AmrDG
//  |
//  NumericalMethod<AmrDG>
//Solver
//  |
//  |
//Model
//  ModelEquation<Euler>
//  |
//Euler

/*
If want to call methods of Modelequn from inside AmrDG, gotta pass template parameter holding COmpressible_Eulr method

*/

//NumericalMethod<AmrDG> stored ptr to AmrDG
template <typename NumericalMethodType,typename ModelEquationType>
Simulation<NumericalMethodType,ModelEquationType>::Simulation() 
{
  //Solver derived class ptr
  num_method = std::make_shared<NumericalMethodType>();
  //Solver base class ptr (implicit conversion)
  //solver = num_method;


  //solver = 
  //std::shared_ptr<Solver<AmrDG>> solverPtr = std::make_shared<AmrDG>();

  //solver->setNumericalMethod(solver);
  //model = std::make_shared<ModelEquationType>();

  //using SolverType = typename std::remove_cvref_t<decltype(SolverTypeTag{})>;
  //using ModelEquationType = typename std::remove_cvref_t<decltype(ModelEquationTypeTag{})>;

  //model->setSolver(solver);
  //model->testModel();
  //model->test();
  //solver->test();
}

template <typename NumericalMethodType,typename ModelEquationType>
Simulation<NumericalMethodType,ModelEquationType>::~Simulation() {
  //ofs->close();
}

template <typename NumericalMethodType,typename ModelEquationType>
void Simulation<NumericalMethodType,ModelEquationType>::run()
{
  //
  //dg_sim->Init();
  //dg_sim->Evolve();
}

#endif // SIMULATION_H

