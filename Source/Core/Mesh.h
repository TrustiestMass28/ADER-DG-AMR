#ifndef MESH_H
#define MESH_H

#include <iostream>
#include <memory>

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Interpolater.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif
#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

using namespace amrex;

template <typename NumericalMethodType>
class Solver;

template <typename NumericalMethodType>
class Mesh : public amrex::AmrCore
{
    public:
        Mesh(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, 
            int _coord, Vector<IntVect> const& _ref_ratios,  
            Array<int,AMREX_SPACEDIM> const& _is_per, 
            amrex::Vector<amrex::Real> _amr_c,
            int _dtn_regrid = 0,amrex::Real _dt_regrid = 0, int _nghost= 1);

        ~Mesh() = default;   

        void init(std::shared_ptr<Solver<NumericalMethodType>> _solver);

        //AmrCore pure virtual function override
        //  Make a new level from scratch using provided BoxArray and DistributionMapping.
        //  Only used during initialization
        virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, 
                                            const amrex::BoxArray& ba,
                                            const amrex::DistributionMapping& dm) override;  
        
        //AmrCore pure virtual function override
        //  Make a new level using provided BoxArray and DistributionMapping and fill with 
        //  interpolated coarse level data.
        virtual void MakeNewLevelFromCoarse(int lev, amrex::Real time,
                                            const amrex::BoxArray& ba, 
                                            const amrex::DistributionMapping& dm) override;

        //AmrCore pure virtual function override
        //  Remake an existing level using provided BoxArray and DistributionMapping and 
        //  fill with existing fine and coarse data.
        virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm) override;

        //AmrCore pure virtual function override
        //  Tags cells for refinement
        virtual void ErrorEst (int lev, amrex::TagBoxArray& tags, 
                                amrex::Real time, int ngrow) override;

        //AmrCore pure virtual function override
        //  Delete level data
        virtual void ClearLevel (int lev) override;

        int get_finest_lev();

        GpuArray<Real, AMREX_SPACEDIM> get_dx(int lev);

        amrex::Real get_dvol(int lev, int d);

        amrex::Real get_dvol(int lev);

        const Vector<Geometry>& get_Geom();// return Geom ();

        const Geometry& get_Geom(int lev);// return Geom (lev);

        IntVect get_refRatio (int lev); 

        const Vector<IntVect>& get_refRatio();

        const amrex::BoxArray& get_BoxArray(int lev) const;

        const amrex::DistributionMapping& get_DistributionMap(int lev) const;

        //Max number of levels
        //max_level 	=	index of max level
        //L 			= 	max number of levels
        //finest_level=	index of curretn finest level

        int L = 1;

        //Regrid time-steps interval
        int dtn_regrid = 0;  

        //Regrid physical time interval
        amrex::Real dt_regrid = 0;  

        //Number of ghost cells
        int nghost= 1;    

        //Refinement criteria coefficient/level multiplier
        amrex::Vector<amrex::Real> amr_c;

    private:
        
        void setSolver(std::shared_ptr<Solver<NumericalMethodType>> _solver);

        std::weak_ptr<Solver<NumericalMethodType>> solver;       
};

template <typename NumericalMethodType>
Mesh<NumericalMethodType>::Mesh(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, 
            int _coord, Vector<IntVect> const& _ref_ratios,  
            Array<int,AMREX_SPACEDIM> const& _is_per, amrex::Vector<amrex::Real> _amr_c,
            int _dtn_regrid, amrex::Real _dt_regrid,int _nghost) 
    :  AmrCore (_rb, _max_level, _n_cell, _coord, _ref_ratios, _is_per) 
{
    L = _max_level+1;
    dtn_regrid = _dtn_regrid;
    dt_regrid = _dt_regrid;
    nghost= _nghost;
    amr_c = _amr_c;
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::init(std::shared_ptr<Solver<NumericalMethodType>> _solver)
{
    setSolver(_solver);

    //TODO: user should specify these formulas in their numerical framework
    SetBlockingFactor(4);
    //SetGridEff(0.9);
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::setSolver(std::shared_ptr<Solver<NumericalMethodType>> _solver)
{
    solver = _solver;
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::MakeNewLevelFromScratch(int lev, amrex::Real time, 
                                    const amrex::BoxArray& ba,
                                    const amrex::DistributionMapping& dm) 
{   
    auto _solver = solver.lock();
    _solver->set_init_data_system(lev, ba, dm);
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::ClearLevel(int lev)
{   
   
    auto _solver = solver.lock();
    _solver->AMR_clear_level_data(lev);   
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::ErrorEst (int lev, amrex::TagBoxArray& tags, 
                                        amrex::Real time, int ngrow)
{
    auto _solver = solver.lock();
    _solver->AMR_tag_cell_refinement(lev,tags,time,ngrow);   
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba,
                        const amrex::DistributionMapping& dm) 
{   
    auto _solver = solver.lock();
    _solver->AMR_remake_level(lev,time,ba,dm);     
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::MakeNewLevelFromCoarse(int lev, amrex::Real time,
                                    const amrex::BoxArray& ba, 
                                    const amrex::DistributionMapping& dm)
{ 
    auto _solver = solver.lock();
    _solver->AMR_make_new_fine_level(lev,time,ba,dm);    
}

template <typename NumericalMethodType>
int Mesh<NumericalMethodType>::get_finest_lev()
{
    return finest_level;
}

template <typename NumericalMethodType>
GpuArray<Real, AMREX_SPACEDIM> Mesh<NumericalMethodType>::get_dx(int lev)
{
    return geom[lev].CellSizeArray();  
}

template <typename NumericalMethodType>
amrex::Real Mesh<NumericalMethodType>::get_dvol(int lev, int d)
{
    //Surface  
    const auto dx = get_dx(lev);

    amrex::Real dvol = 1.0;
    for(int _d = 0; _d < AMREX_SPACEDIM; ++_d){
      if(_d!=d){dvol*=dx[_d];}
    }

    return dvol;
}

template <typename NumericalMethodType>
amrex::Real Mesh<NumericalMethodType>::get_dvol(int lev)
{
    //Volume
    const auto dx = get_dx(lev);

    amrex::Real dvol = 1.0;
    for(int _d = 0; _d < AMREX_SPACEDIM; ++_d){
      dvol*=dx[_d];
    }

    return dvol;
} 


template <typename NumericalMethodType>
const Vector<Geometry>& Mesh<NumericalMethodType>::get_Geom()
{
    return Geom();
}
 
template <typename NumericalMethodType>
const Geometry& Mesh<NumericalMethodType>::get_Geom(int lev)
{
    return Geom(lev);
}

template <typename NumericalMethodType>
IntVect  Mesh<NumericalMethodType>::get_refRatio (int lev)
{
    return refRatio(lev);
} 

template <typename NumericalMethodType>
const Vector<IntVect>&  Mesh<NumericalMethodType>::get_refRatio()
{
    return refRatio();
}

template <typename NumericalMethodType>
const amrex::BoxArray& Mesh<NumericalMethodType>::get_BoxArray(int lev) const
{
    return boxArray(lev);
}

template <typename NumericalMethodType>
const amrex::DistributionMapping& Mesh<NumericalMethodType>::get_DistributionMap(int lev) const
{
    return DistributionMap(lev);
}
#endif 