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

class ModelEquation;
class NumericalMethod;

class Simulation
{
  public:    
    Simulation(ModelEquation* _model, NumericalMethod* _method);
    
    ~Simulation();

    void run();
    
  private:
    int _coord = 0;//cartesian, don't touch

    std::shared_ptr<std::ofstream> ofs;

    std::shared_ptr<ModelEquation> model_pde;

    std::shared_ptr<NumericalMethod> numerical_pde;

};

#endif // SIMULATION_H