#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>
#include <cmath>
#include <math.h>
#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

#include <string>
#include <fstream>
#include <iostream>

#include "Simulation.h"
//#include "ModelEquation.h"
#include "Solver.h"

using namespace amrex;

template <typename SolverType> //typename ModelEquationType, 
Simulation<SolverType>::Simulation(SolverType* _method) :  solver(_method) //ModelEquationType* _model, model_pde(_model) ,
{
  //Exchange classes ptrs s.t they can communicate

  //use numerical_pde to call method of NumericalMethod
  //this method is actually virtual and overridden by implementation of AmrDG
  //therefore the AmrDG implementation will be called

  //model_pde->setNumericalMethod(numerical_pde);
  //solver->setModelEquation(model_pde);

  //ofs->open("simulation_output.txt", std::ofstream::out);
  //model_pde->setOfstream(ofs);
  //numerical_pde->setOfstream(ofs);
}

template <typename SolverType>
Simulation<SolverType>::~Simulation() {
  //ofs->close();
}

template <typename SolverType>
void Simulation<SolverType>::run()
{
  //dg_sim->Init();
  //dg_sim->Evolve();
}