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
            Array<int,AMREX_SPACEDIM> const& _is_per, int _dtn_regrid = 0, 
            int _dt_regrid = 0, int _nghost= 1);

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

        //Max number of levels
        int L = 1;

        //Regrid time-steps interval
        int dtn_regrid = 0;  

        //Regrid physical time interval
        int dt_regrid = 0;  

        //Number of ghost cells
        int nghost= 1;    

    private:
        
        void setSolver(std::shared_ptr<Solver<NumericalMethodType>> _solver);

        std::shared_ptr<Solver<NumericalMethodType>> solver;       
};

template <typename NumericalMethodType>
Mesh<NumericalMethodType>::Mesh(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, 
            int _coord, Vector<IntVect> const& _ref_ratios,  
            Array<int,AMREX_SPACEDIM> const& _is_per, int _dtn_regrid , 
            int _dt_regrid,int _nghost) 
    :  AmrCore (_rb, _max_level, _n_cell, _coord, _ref_ratios, _is_per) 
{
    L = _max_level+1;
    dtn_regrid = _dtn_regrid;
    dt_regrid = _dt_regrid;
    nghost= _nghost;
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::init(std::shared_ptr<Solver<NumericalMethodType>> _solver)
{
    setSolver(solver);
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
    solver->set_init_data_system(lev, ba, dm);
}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::MakeNewLevelFromCoarse(int lev, amrex::Real time,
                                    const amrex::BoxArray& ba, 
                                    const amrex::DistributionMapping& dm) {}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba,
                        const amrex::DistributionMapping& dm) {}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::ErrorEst (int lev, amrex::TagBoxArray& tags, 
                    amrex::Real time, int ngrow) {}

template <typename NumericalMethodType>
void Mesh<NumericalMethodType>::ClearLevel (int lev) {}

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


/*
ADAPTIVE MESH REFINEMENT (GEOMETRY BASED OPERATIONS)
    //AMR settings 




    void FillPatch (int lev, Real time, amrex::MultiFab& mf, int icomp, int ncomp, int q);
    
    void FillCoarsePatch (int lev, Real time, amrex::MultiFab& mf, int icomp, int ncomp, int q);
    
    void GetData (int lev, int q, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime);
    
    void AverageFineToCoarse();    
    
    void AverageFineToCoarseFlux(int lev);
    
    void FillPatchGhostFC(int lev,amrex::Real time,int q);

    //AMR refinement and limiting
    void AMR_settings_tune();

AMR INTERPOLATOR


 
    //AMR Coarse>->Fine projection custom implementation
    class DGprojInterp : public Interpolater
    {
      public:
        
        Box CoarseBox (const Box& fine, int ratio) override;
        
        Box CoarseBox (const Box& fine, const IntVect& ratio) override;
            
        void interp (const FArrayBox& crse, int crse_comp,FArrayBox& fine,
                    int  fine_comp,int ncomp, const Box& fine_region, 
                    const IntVect&   ratio, const Geometry& crse_geom, 
                    const Geometry& fine_geom,Vector<BCRec> const& bcr,
                    int actual_comp,int actual_state, RunOn runon) override;
                           
        void amr_scatter(int i, int j, int k, int n, Array4<Real> const& fine, 
                        int fcomp, Array4<Real const> const& crse, int ccomp, 
                        int ncomp, IntVect const& ratio) noexcept;
                                            
        void average_down(const MultiFab& S_fine, MultiFab& S_crse,
                         int scomp, int ncomp, const IntVect& ratio, const int lev_fine, 
                         const int lev_coarse, int d=0, bool flag_flux=false);
                  
        void amr_gather(int i, int j, int k, int n,Array4<Real> const& crse, 
                        Array4<Real const> const& fine,int ccomp, 
                        int fcomp, IntVect const& ratio) noexcept;
                     
        //void amr_gather_flux(int i, int j, int k, int n, int d,Array4<Real> const& crse, 
        //                                  Array4<Real const> const& fine,int ccomp, 
        //                                  int fcomp, IntVect const& ratio) noexcept;                               
        void getouterref(AmrDG* _amrdg);  
        
        void interp_proj_mat();
        
        void average_down_flux(MultiFab& S_fine, MultiFab& S_crse,
                                      int scomp, int ncomp, const IntVect& ratio, 
                                      const int lev_fine, const int lev_coarse, 
                                      int d, bool flag_flux);
                                      
         void amr_gather_flux(int i, int j, int k, int n, int d,Array4<Real> const& crse, 
                                          Array4<Real> const& fine,int ccomp, 
                                          int fcomp, IntVect const& ratio) noexcept;

      //amrex::Real RefMat_phiphi(int i,int j, bool is_predictor, bool is_mixed_nmodes) const;

      private:     
        friend class NumericalMethod;
          
        AmrDG* amrdg;     
        
        amrex::Vector<amrex::Vector<int >> amr_projmat_int;
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> P_scatter;
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> P_gather;        
    };



*/

#endif 