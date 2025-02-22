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

class Mesh : public amrex::AmrCore
{
    public:
        Mesh(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, 
            int _coord, Vector<IntVect> const& _ref_ratios,  
            Array<int,AMREX_SPACEDIM> const& _is_per, int _L = 1, int _dtn_regrid = 0, 
            int _dt_regrid = 0, int _nghost= 1);

        ~Mesh() = default;   

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

        //Max number of levels
        int L = 1;

        //Regrid time-steps interval
        int dtn_regrid = 0;  

        //Regrid physical time interval
        int dt_regrid = 0;  

        //Number of ghost cells
        int nghost= 1;    
};

Mesh::Mesh(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, 
            int _coord, Vector<IntVect> const& _ref_ratios,  
            Array<int,AMREX_SPACEDIM> const& _is_per, int _L, int _dtn_regrid , 
            int _dt_regrid,int _nghost) 
    :  AmrCore (_rb, _max_level, _n_cell, _coord, _ref_ratios, _is_per) 
{
    L = _L;
    dtn_regrid = _dtn_regrid;
    dt_regrid = _dt_regrid;
    nghost= _nghost;
}

void Mesh::MakeNewLevelFromScratch(int lev, amrex::Real time, 
                                    const amrex::BoxArray& ba,
                                    const amrex::DistributionMapping& dm) {}

void Mesh::MakeNewLevelFromCoarse(int lev, amrex::Real time,
                                    const amrex::BoxArray& ba, 
                                    const amrex::DistributionMapping& dm) {}

void Mesh::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba,
                        const amrex::DistributionMapping& dm) {}

void Mesh::ErrorEst (int lev, amrex::TagBoxArray& tags, 
                    amrex::Real time, int ngrow) {}

void Mesh::ClearLevel (int lev) {}


#endif 