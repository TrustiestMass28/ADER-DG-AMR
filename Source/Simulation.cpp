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
#include "ModelEquation.h"
#include "NumericalMethod.h"

using namespace amrex;

Simulation::Simulation(ModelEquation* _model, NumericalMethod* _method) : model_pde(_model) , numerical_pde(_method)
{
  //Exchange classes ptrs s.t they can communicate


  //use numerical_pde to call method of NumericalMethod
  //this method is actually virtual and overridden by implementation of AmrDG
  //therefore the AmrDG implementation will be called

  model_pde->setNumericalMethod(numerical_pde);
  numerical_pde->setModelEquation(model_pde);

  ofs->open("simulation_output.txt", std::ofstream::out);
  model_pde->setOfstream(ofs);
  numerical_pde->setOfstream(ofs);
}

Simulation::~Simulation() {
  ofs->close();
}

void Simulation::run()
{
  //dg_sim->Init();
  //dg_sim->Evolve();
}