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
#include "Mesh.h"

using namespace amrex;

template <typename NumericalMethodType,typename EquationType>
class Simulation
{
  public:    
    Simulation(); 
    
    ~Simulation();

    void run();

    template <typename... Args>
    void setNumericalSettings(Args... args);
    
    template <typename... Args>
    void setModelSettings(Args... args);

    void setGeometrySettings(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, 
                    int _coord, Vector<IntVect> const& _ref_ratios,  
                    Array<int,AMREX_SPACEDIM> const& _is_per, int L = 1, int dtn_regrid = 0, 
                    int dt_regrid = 0,int nghost= 1);

    void setIO(int _n_out, amrex::Real _t_out);

  private:
    int _coord = 0;//cartesian, don't touch

    std::shared_ptr<std::ofstream> ofs;

    std::shared_ptr<ModelEquation<EquationType>> model;

    std::shared_ptr<Solver<NumericalMethodType>> solver;

    std::shared_ptr<Mesh<NumericalMethodType>> mesh;

    //I/O 
    int dtn_outplt;   //data output time-steps interval

    amrex::Real dt_outplt;   //data output physical time interval
};


template <typename NumericalMethodType,typename EquationType>
Simulation<NumericalMethodType,EquationType>::Simulation() 
{}

template <typename NumericalMethodType,typename EquationType>
Simulation<NumericalMethodType,EquationType>::~Simulation() {
  //ofs->close();
}

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::run()
{
  solver->init(model,mesh);
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
void Simulation<NumericalMethodType,EquationType>::setGeometrySettings(const RealBox& _rb, int _max_level,
                                                                      const Vector<int>& _n_cell, 
                                                                      int _coord, Vector<IntVect> const& _ref_ratios,  
                                                                      Array<int,AMREX_SPACEDIM> const& _is_per,
                                                                      int L , int dtn_regrid , 
                                                                      int dt_regrid ,int nghost)
{
  mesh = std::make_shared<Mesh<NumericalMethodType>>(_rb,_max_level,_n_cell,_coord,_ref_ratios,_is_per,
                    L, dtn_regrid, dt_regrid, nghost);   
} 

template <typename NumericalMethodType,typename EquationType>
void Simulation<NumericalMethodType,EquationType>::setIO(int _n_out, amrex::Real _t_out)
{
  dtn_outplt = _n_out;
  dt_outplt  = _t_out;
}

#endif // SIMULATION_H

