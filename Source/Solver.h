#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <memory>

#include <AMReX_AmrCore.H>
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

#include <indicators/progress_bar.hpp>
#include <sstream>
#include <iomanip>

template <typename NumericalMethodType>
class Mesh;

template <typename EquationType,typename NumericalMethodType>
class BoundaryCondition;

template <typename EquationType>
class ModelEquation;

using namespace amrex;

template <typename NumericalMethodType>
class Solver
{
    public: 

        Solver() = default;

        virtual ~Solver() = default;

        template <typename... Args>
        void settings(Args... args) {
            static_cast<NumericalMethodType*>(this)->settings(std::forward<Args>(args)...);
        }

        //TODO: pass all tempaltes of other classes from which Solver might need data to init
        //like stuff from geometry for number of levels,...
        template <typename EquationType>
        void init(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                    std::shared_ptr<Mesh<NumericalMethodType>> _mesh,
                    int _dtn_outplt, amrex::Real _dt_outplt, std::string _out_name_prefix);

        //reshape bc vector depending on solver used (e.g if use modal or not)
        void init_bc(amrex::Vector<amrex::Vector<amrex::BCRec>>& bc, int& n_comp);

        //execute simulation (time-stepping and possible AMR operations)
        template <typename EquationType>
        void evolve(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                    const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond);

        //perform a time-step, advance solution by one time-step
        template <typename EquationType>
        void time_integration(const  std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                              const  std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                                amrex::Real time);

        //compute and set time-step size
        template <typename EquationType>
        void set_Dt(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);

        //get solution vector evaluation
        void get_U();

        //reconstruct solution vector evaluation based on its decomposition
        //and quadrature. M,N to be specified s.t we cna choose
        //to evaluate at boundary or in cell
        void get_U_from_U_w(int M, int N,amrex::Vector<amrex::MultiFab>* U_ptr,
                            amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                            const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

        template <typename EquationType>
        void source(int lev,int M, 
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::Vector<amrex::MultiFab>* U_ptr,
                    amrex::Vector<amrex::MultiFab>* S_ptr,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

        template <typename EquationType>                
        void flux(int lev,int d, int M, 
                    const std::shared_ptr<ModelEquation<EquationType>>&  model_pde,
                    amrex::Vector<amrex::MultiFab>* U_ptr,
                    amrex::Vector<amrex::MultiFab>* F_ptr,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

        template <typename EquationType>
        void flux_bd(int lev,int d, int M,
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::Vector<amrex::MultiFab>* U_ptr,
                    amrex::Vector<amrex::MultiFab>* F_ptr,
                    amrex::Vector<amrex::MultiFab>* DF_ptr,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

        void numflux(int lev,int d,int M, int N,
                    amrex::Vector<amrex::MultiFab>* U_ptr_m, 
                    amrex::Vector<amrex::MultiFab>* U_ptr_p,
                    amrex::Vector<amrex::MultiFab>* F_ptr_m,
                    amrex::Vector<amrex::MultiFab>* F_ptr_p,
                    amrex::Vector<amrex::MultiFab>* DF_ptr_m,
                    amrex::Vector<amrex::MultiFab>* DF_ptr_p);

        //computes the integral of the numerical flux across interface
        void numflux_integral(int lev,int d,int M, int N,
                            amrex::Vector<amrex::MultiFab>* U_ptr_m, 
                            amrex::Vector<amrex::MultiFab>* U_ptr_p,
                            amrex::Vector<amrex::MultiFab>* F_ptr_m,
                            amrex::Vector<amrex::MultiFab>* F_ptr_p,
                            amrex::Vector<amrex::MultiFab>* DF_ptr_m,
                            amrex::Vector<amrex::MultiFab>* DF_ptr_p);

        //get solution vector derivative
        template <typename EquationType>
        void get_dU(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);

        //sometimes IC is for modes/weights, other for actual sol vector
        //depending on num method it will call 
        template <typename EquationType>
        void set_initial_condition(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);
        
        //declare and init data structures holding system equation 
        void set_init_data_system(int lev,const BoxArray& ba,
                                    const DistributionMapping& dm);

        //declare and init data structures holding single equation 
        void set_init_data_component(int lev,const BoxArray& ba, 
                                        const DistributionMapping& dm, int q);

        //Average fine to coarse the initial condition, will generally call existing
        //gathering/average down method required for interpolation and sync between timesteps
        //void AMR_avg_down_initial_condition();

        void AMR_advanced_settings();

        //Scatter IC from coarse to fine levels, used only during initialiation
        //user provide it sown implementation s.t cna specify which MFab required
        //projection
        void AMR_interpolate_initial_condition(int lev);

        void AMR_average_fine_coarse();

        //clear all data of the MFabs at specified level
        void AMR_clear_level_data(int lev);

        //tag cells for refinement based on implemented metric
        void AMR_tag_cell_refinement(int lev, amrex::TagBoxArray& tags, 
                                    amrex::Real time, int ngrow);


        void AMR_remake_level(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm);

        void AMR_make_new_fine_level(int lev, amrex::Real time,
                                    const amrex::BoxArray& ba, 
                                    const amrex::DistributionMapping& dm);
        
        void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
            ofs = _ofs;
        }

        template <typename EquationType>
        void PlotFile(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                                    amrex::Vector<amrex::Vector<amrex::MultiFab>>& X,
                                                    int tstep, amrex::Real time) const;
                                                
        //Apply boundary conditions by calling BC methods
        template <typename EquationType>
        void FillBoundaryCells(const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                                amrex::Vector<amrex::MultiFab>* U_ptr, 
                                int lev, amrex::Real time);


        void FillBoundary(amrex::MultiFab* U_ptr, int lev);

        //Transformations to BC cell vlaues before being set as final value
        amrex::Real setBC(const amrex::Vector<amrex::Real>& bc, int comp,int dcomp,int q, int lev);

        //Creat a dummy BC vector, usefull in palces of the code where cannot access
        //custom boundary object and don't want to apply BC, but amrex functions require to
        //pass a BC object
        amrex::Vector<amrex::Vector<amrex::BCRec>> get_null_BC_vct(int ncomp, int q);

        amrex::Vector<amrex::BCRec> get_null_BC(int ncomp);

        //General class for numerical methods that use basis decomposition of the solution
        //can maange spatial,temporal and mixed basis functions
        //TODO: use CRTP
        class Basis{
            public:
                Basis() = default;

                ~Basis();

                //Spatial basis function, evaluated at x
                //NB: dim(x) = AMREX_SPACEDIM
                virtual amrex::Real phi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map, 
                                            const amrex::Vector<amrex::Real>& x) const { return 0.0; }

                //Spatial basis function first derivative dphi/dx_d, evaluated at x
                virtual amrex::Real dphi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                                            const amrex::Vector<amrex::Real>& x, int d) const { return 0.0; }

                //Spatial basis function second derivative d^2phi/dx_d1dx_d2, evaluated at x
                virtual amrex::Real ddphi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                                            const amrex::Vector<amrex::Real>& x, int d1, int d2) const { return 0.0; }

                //Temporal basis function, evaluated at t
                //NB: dim(t) = 1
                virtual amrex::Real phi_t(int tidx, amrex::Real tau) const { return 0.0; }

                //Temporal basis function first derivative dtphi/dt, evaluated at t
                virtual amrex::Real dtphi_t(int tidx, amrex::Real tau) const { return 0.0; }

                //Spatio-temporal basis function, evaluated at x
                //NB: dim(x) = AMREX_SPACEDIM+1
                virtual amrex::Real phi_st(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                                            const amrex::Vector<amrex::Real>& x) const { return 0.0; }

                //First derivative
                virtual amrex::Real dphi_st(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                                            const amrex::Vector<amrex::Real>& x) const { return 0.0; }

                //Second derivative
                virtual amrex::Real ddphi_st(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                                        const amrex::Vector<amrex::Real>& x) const { return 0.0; }

                //Set number of basis function/weights/modes Np,mNp
                virtual void set_number_basis() {}

                //Set spatial basis functions Phi(x) index mapping
                virtual void set_idx_mapping_s() {}

                //Set temporal basis function Phi(t) index mapping
                virtual void set_idx_mapping_t() {}

                //Set spatio-temporal basis functions Phi(x,t) index mapping
                virtual void set_idx_mapping_st() {}

                //Number of spatial basis functions/modes
                int Np_s; 

                //Number of temporal basis functions/modes
                int Np_t; 

                //Number of spatio-temporal basis functions/modes
                int Np_st; 

                //Spatial basis functions Phi(x) index mapping
                amrex::Vector<amrex::Vector<int>> basis_idx_s; 
                //  used to store the combinations of indices of 1d Basis functions: e.g
                //  basis_idx[5] = [0,1,4] ==> phi_5= P_0*P_1*P_4
                //  with P_i e.g beign the i-th Legendre polynomial 1d basis

                //Set temporal basis function Phi(t) index mapping
                amrex::Vector<amrex::Vector<int>> basis_idx_t;

                //Set spatio-temporal basis functions Phi(x,t) index mapping
                amrex::Vector<amrex::Vector<int>> basis_idx_st;

                void setNumericalMethod(std::shared_ptr<NumericalMethodType> _numme);

            protected:
                //Ptr used to access numerical method and solver data
                //NumericalMethodType* numme;
                std::shared_ptr<NumericalMethodType> numme;

        };

        class Quadrature{
            public:
                Quadrature() = default;

                ~Quadrature();

                //Set number of quadrature points (should be func of order p)
                virtual void set_number_quadpoints() {};

                //Generate the quadrature points
                virtual void set_quadpoints() {};

                //Interpolation nodes/quadrature points
                //  for spatial basis functions
                amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_quad_s;
                //  for spatial basis function at the boundaries
                amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> xi_ref_quad_s_bdm;

                amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> xi_ref_quad_s_bdp;

                //  for single cell center evaluation , to be sued e.g for Dt computation
                amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_quad_s_center;
                   
                //  for temporal basis functions
                amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_quad_t;

                //  for spatio-temporal basis functions
                amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_quad_st;
                //  for spatio-temporal basis function at the boundaries
                amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> xi_ref_quad_st_bdm;

                amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> xi_ref_quad_st_bdp;

                //amrex::Vector<amrex::Vector<amrex::Real>> L2proj_quadmat;
                //amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_GLquad_L2proj;
                // amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_equidistant; 

                //number of quadrature points in 1 dimensios, used to have 
                //same amount of points per dimension
                int qMp_1d;    

                //number of quadrature points for quadrature of volume (spatio) integral   
                int qMp_s;    
                //  at the boundary, surface (spatio) integral
                int qMp_s_bd;   

                //number of quadrature points for quadrature of volume (temporal) integral   
                int qMp_t;  

                //number of quadrature points for quadrature of volume (spatio-temporal) integral   
                int qMp_st;    
                //  at the boundary, surface (spatio-temporal) integral
                int qMp_st_bd;     

                //  int qMp_L2proj; //number of quadrature points only in space, used for the BCs,ICs,

                void setNumericalMethod(std::shared_ptr<NumericalMethodType> _numme);
                
                protected:
                    //Ptr used to access numerical method and solver data
                    std::shared_ptr<NumericalMethodType> numme;
            };

            template <typename InterpolationType>
            class AMR_Interpolation : public amrex::Interpolater
            {
                public:
                    AMR_Interpolation() = default;

                    virtual ~AMR_Interpolation() = default; // Good practice

                    //Override pure virtual function from Interpolater
                    void interp (const FArrayBox& crse,
                                int              crse_comp,
                                FArrayBox&       fine,
                                int              fine_comp,
                                int              ncomp,
                                const Box&       fine_region,
                                const IntVect&   ratio,
                                const Geometry&  crse_geom,
                                const Geometry&  fine_geom,
                                Vector<BCRec> const& bcr,
                                int              actual_comp,
                                int              actual_state,
                                RunOn            runon) override;

                    Box CoarseBox (const Box& fine, int ratio) override;


                    Box CoarseBox (const Box& fine, const IntVect& ratio) override;

                    
                    void setNumericalMethod(std::shared_ptr<NumericalMethodType> _numme);

                protected:
                    //Ptr used to access numerical method and solver data
                    std::shared_ptr<NumericalMethodType> numme;
            };

            //number of points at which boundary conditions are
            //need to be set by the NumericalMethodType, called
            //by BoundaryCondition. Set in init_bc
            int n_pt_bc;
            
    protected:
        std::shared_ptr<std::ofstream> ofs;

        std::weak_ptr<Mesh<NumericalMethodType>> mesh;

        //number of equations of the system
        int Q; 

        //number of lin-indep equations of the system which are non
        //derivable from others, i.e number of solution unknowns
        //which are independent/unique/not function of others (e.g 
        //not the angular momentum)
        int Q_unique; 

        //Flag to indicate if source term is considered
        bool flag_source_term;  

        //spatial (approxiamtion) order
        int p;

        //Courant–Friedrichs–Lewy number
        amrex::Real CFL;

        //Time step size
        amrex::Real Dt;

        //Physical simulated time
        amrex::Real T;

        //I/O 
        int dtn_outplt;   //data output time-steps interval
        amrex::Real dt_outplt;   //data output physical time interval

        std::string out_name_prefix;

        //Multifabs vectors (LxDxQ or LxQ)
        //L:  max number of levels
        //D:  dimensions
        //Q:  number of solution components

        //solution vector U(x,t) 
        amrex::Vector<amrex::Vector<amrex::MultiFab>> U; 

        //Modal/Nodal solution vector U_w
        amrex::Vector<amrex::Vector<amrex::MultiFab>> U_w;

        //solution vector U(x,t) evalauted at cells center
        amrex::Vector<amrex::Vector<amrex::MultiFab>> U_center;

        //Physical flux F(x,t)
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> F;
        //  derivative
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> DF;

        //Physical flux F(x,t) evaluated at boundary minus (-) b-
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> Fm;
        //  derivative
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> DFm;

        //Physical flux F(x,t) evaluated at boundary plus (+) b+
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> Fp;
        //  derivative
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> DFp;

        //Numerical flux approximation Fnum(x,t)
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> Fnum;

        //Numerical flux approximation Fnum(x,t) integrated over boundary minus (-) b-
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> Fnumm_int; 

        //Numerical flux approximation Fnum(x,t) integrated over boundary plus (+) b+
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> Fnump_int;

        //Source/Sink term S(x,t)
        amrex::Vector<amrex::Vector<amrex::MultiFab>> S;

        std::optional<indicators::ProgressBar> m_bar;

    private:
    
        void setMesh(std::shared_ptr<Mesh<NumericalMethodType>> _mesh);

};

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_advanced_settings()
{
     static_cast<NumericalMethodType*>(this)->AMR_advanced_settings();
}

template <typename NumericalMethodType>
amrex::Vector<amrex::BCRec> Solver<NumericalMethodType>::get_null_BC(int ncomp)
{
    amrex::Vector<amrex::BCRec> _bc(ncomp);
    for (int n = 0; n < ncomp; ++n) {
        _bc[n].setLo(AMREX_D_DECL(amrex::BCType::int_dir,
                                amrex::BCType::int_dir,
                                amrex::BCType::int_dir));

        _bc[n].setHi(AMREX_D_DECL(amrex::BCType::int_dir,
                                amrex::BCType::int_dir,
                                amrex::BCType::int_dir));
    }
   
    return _bc;
}
  
template <typename NumericalMethodType>
amrex::Vector<amrex::Vector<amrex::BCRec>> Solver<NumericalMethodType>::get_null_BC_vct(int ncomp, int q)
{
    amrex::Vector<amrex::Vector<amrex::BCRec>> _bc(q, amrex::Vector<amrex::BCRec> (ncomp));

    for(int i=0 ; i<Q; ++i){
        _bc[i] = get_null_BC(ncomp);
    }

    return _bc;
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::init( const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                                        std::shared_ptr<Mesh<NumericalMethodType>> _mesh,
                                        int _dtn_outplt, amrex::Real _dt_outplt, std::string _out_name_prefix) 
{
    //set I/O
    dtn_outplt = _dtn_outplt;
    dt_outplt = _dt_outplt;
    out_name_prefix = _out_name_prefix;

    setMesh(_mesh);
    
    //Get model specific data that influence numerical set-up
    Q = model_pde->Q_model;
    Q_unique = model_pde->Q_model_unique;
    flag_source_term = model_pde->flag_source_term;

    //Numerical method specific initialization
    static_cast<NumericalMethodType*>(this)->init();

    const Real time = 0.0;

	//Init progress bar
    if (amrex::ParallelDescriptor::IOProcessor()) {
        m_bar.emplace(
            indicators::option::BarWidth{90},
            indicators::option::Start{"["},
            indicators::option::Fill{"█"},
            indicators::option::Lead{"█"},
            indicators::option::Remainder{"░"},
            indicators::option::End{"]"},
            indicators::option::PostfixText{"Initializing..."},
            indicators::option::ForegroundColor{indicators::Color::yellow},
            indicators::option::FontStyles{
                std::vector<indicators::FontStyle>{indicators::FontStyle::bold}
            },
            indicators::option::Stream{std::cout},  //CRITICAL for in-place update
            indicators::option::MaxProgress{100} // REQUIRED for set_progress(0–100)
        );
    }

    //AmrCore.h function initialize multilevel mesh, geometry, Box array and DistributionMap
    //calls MakeNewLevelFromScratch
    //Initialize BoxArray, DistributionMapping and data from scratch.
    //Calling this function requires the derive class implement its own MakeNewLevelFromScratch
    //to allocate and initialize data.

    //NB: Construct full domain grids up to max_level
    _mesh->InitFromScratch(time);

    //Populate all grid levels with the IC
    //afterwards apply regridding to tag cells
    //and create AMR grids
    set_initial_condition(model_pde);
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::init_bc(amrex::Vector<amrex::Vector<amrex::BCRec>>& bc, int& n_comp)
{   
    static_cast<NumericalMethodType*>(this)->init_bc(bc, n_comp); 
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::setMesh(std::shared_ptr<Mesh<NumericalMethodType>> _mesh)
{
    mesh = _mesh;
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_clear_level_data(int lev)
{
    static_cast<NumericalMethodType*>(this)->AMR_clear_level_data(lev);
}
template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_tag_cell_refinement(int lev, amrex::TagBoxArray& tags, 
                                                        amrex::Real time, int ngrow)
{
    static_cast<NumericalMethodType*>(this)->AMR_tag_cell_refinement(lev,tags,time,ngrow);
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_make_new_fine_level(int lev, amrex::Real time,
                                                        const amrex::BoxArray& ba, 
                                                        const amrex::DistributionMapping& dm)
{
    static_cast<NumericalMethodType*>(this)->AMR_make_new_fine_level(lev,time,ba,dm);    
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::set_init_data_system(int lev,const BoxArray& ba,
                                                        const DistributionMapping& dm)
{
    //NumericalMethod specific data structure initialization (e.g additional)
    //can also clear up Solver data members that arent needed for particular method
    //e.g the numerical fluxes

    static_cast<NumericalMethodType*>(this)->set_init_data_system(lev, ba, dm);
    
    //init data for each ocmponent of the equation
    for(int q=0; q<Q; ++q){
        set_init_data_component(lev,ba,dm, q);
    }
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::set_init_data_component(int lev,const BoxArray& ba, 
                                                        const DistributionMapping& dm, int q)
{   
    static_cast<NumericalMethodType*>(this)->set_init_data_component(lev, ba, dm, q);
}

/*
    //Initialize al finer levels with full grid solution interpolated form coarsest
    //This is needed because at start-up of time-stepping, regrid will be performed
    //and since all fine levels are currently covering all domain, after regrid
    //the new regridded fine levelswill still overlap with their pre-regrid grid
    //therefore data will be jsut copied and if these fine levels 
    //with started un-initialized IC, they will remain un-initialized.
    //For this reason need to populate them befrehand
    
    //Since the current grids at all levels cover the entire domain, 
    //when regridding at 0th timestep
    //no projection/interpolation will be made
*/
template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::set_initial_condition(const std::shared_ptr<ModelEquation<EquationType>>& model_pde)
{
    auto _mesh = mesh.lock();
   
    //Define IC on single coarse mesh
    static_cast<NumericalMethodType*>(this)->set_initial_condition(model_pde,0);
    
    if (_mesh->L > 1) {
        for(int l=1; l<_mesh->L; ++l){
            AMR_interpolate_initial_condition(l);
            //static_cast<NumericalMethodType*>(this)->set_initial_condition(model_pde,l);
        }
    }
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_interpolate_initial_condition(int lev)
{
    static_cast<NumericalMethodType*>(this)->AMR_interpolate_initial_condition(lev);    
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_average_fine_coarse()
{
    static_cast<NumericalMethodType*>(this)->AMR_average_fine_coarse();    
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_remake_level(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                                    const amrex::DistributionMapping& dm) 
{
   static_cast<NumericalMethodType*>(this)->AMR_remake_level(lev,time,ba,dm);
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::Basis::setNumericalMethod(std::shared_ptr<NumericalMethodType> _numme)
{
    numme = _numme;
}

template <typename NumericalMethodType>
Solver<NumericalMethodType>::Basis::~Basis()
{
    //delete numme;
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::Quadrature::setNumericalMethod(std::shared_ptr<NumericalMethodType> _numme)
{
    numme = _numme;
}

template <typename NumericalMethodType>
Solver<NumericalMethodType>::Quadrature::~Quadrature()
{
    //delete numme;
}

template <typename NumericalMethodType>
template <typename InterpolationType>
void Solver<NumericalMethodType>::AMR_Interpolation<InterpolationType>::setNumericalMethod(std::shared_ptr<NumericalMethodType> _numme)
{
    numme = _numme;
}

template <typename NumericalMethodType>
template <typename InterpolationType>
void Solver<NumericalMethodType>::AMR_Interpolation<InterpolationType>::interp(const FArrayBox& crse,
                                                                                int              crse_comp,
                                                                                FArrayBox&        fine,
                                                                                int              fine_comp,
                                                                                int              ncomp,
                                                                                const Box&        fine_region,
                                                                                const IntVect&    ratio,
                                                                                const Geometry&   crse_geom,
                                                                                const Geometry&   fine_geom,
                                                                                Vector<BCRec> const& bcr,
                                                                                int              actual_comp,
                                                                                int              actual_state,
                                                                                RunOn             runon)
{
    static_cast<InterpolationType*>(this)->interp(crse, crse_comp, fine, fine_comp,
                                                    ncomp, fine_region, ratio,
                                                    crse_geom, fine_geom, bcr,
                                                    actual_comp, actual_state, runon);
}

template <typename NumericalMethodType>
template <typename InterpolationType>
Box Solver<NumericalMethodType>::AMR_Interpolation<InterpolationType>::CoarseBox (const Box& fine, int ratio) 
{
    return static_cast<InterpolationType*>(this)->CoarseBox(fine, ratio);
}

template <typename NumericalMethodType>
template <typename InterpolationType>
Box Solver<NumericalMethodType>::AMR_Interpolation<InterpolationType>::CoarseBox (const Box& fine, const IntVect& ratio) 
{
    return static_cast<InterpolationType*>(this)->CoarseBox(fine, ratio);
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::evolve(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond)
{
    static_cast<NumericalMethodType*>(this)->evolve(model_pde,bdcond); 
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::FillBoundaryCells(const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                                                    amrex::Vector<amrex::MultiFab>* U_ptr, 
                                                    int lev, amrex::Real time)

{
    //static_cast<NumericalMethodType*>(this)->FillBoundaryCells(mesh,bdcond,U_ptr,lev,time); 
    //TODO: in case AmrDG should prvide some specialized functionalities to BC maybe then static_cast is needed
    bdcond->FillBoundaryCells(U_ptr, lev, time);
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::FillBoundary(amrex::MultiFab* U_ptr,int lev)
{
    auto _mesh = mesh.lock();
    amrex::Geometry geom_l = _mesh->get_Geom(lev);

    U_ptr->FillBoundary(geom_l.periodicity());
}

template <typename NumericalMethodType>
amrex::Real Solver<NumericalMethodType>::setBC(const amrex::Vector<amrex::Real>& bc, int comp,int dcomp,int q, int lev){

    return static_cast<NumericalMethodType*>(this)->setBC(bc,comp,dcomp,q,lev); 

}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::time_integration(const std::shared_ptr<ModelEquation<EquationType>>&  model_pde, 
                                                   const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                                                    amrex::Real time)
{
    static_cast<NumericalMethodType*>(this)->time_integration(model_pde,bdcond,time); 
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::set_Dt(const std::shared_ptr<ModelEquation<EquationType>>& model_pde)
{
    static_cast<NumericalMethodType*>(this)->set_Dt(model_pde);     
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::get_U_from_U_w(int M, int N,amrex::Vector<amrex::MultiFab>* U_ptr,
                                                amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                                                const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    static_cast<NumericalMethodType*>(this)->get_U_from_U_w(M,N,U_ptr,U_w_ptr,xi);    
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::source(int lev,int M, 
                                        const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        amrex::Vector<amrex::MultiFab>* U_ptr,
                                        amrex::Vector<amrex::MultiFab>* S_ptr,
                                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    static_cast<NumericalMethodType*>(this)->source(lev,M,model_pde,U_ptr,S_ptr,xi); 
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::flux(int lev,int d, int M, 
                                        const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        amrex::Vector<amrex::MultiFab>* U_ptr,
                                        amrex::Vector<amrex::MultiFab>* F_ptr,
                                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    static_cast<NumericalMethodType*>(this)->flux(lev,d,M,model_pde,U_ptr,F_ptr,xi); 
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::flux_bd(int lev,int d, int M,
                                        const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        amrex::Vector<amrex::MultiFab>* U_ptr,
                                        amrex::Vector<amrex::MultiFab>* F_ptr,
                                        amrex::Vector<amrex::MultiFab>* DF_ptr,
                                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    static_cast<NumericalMethodType*>(this)->flux_bd(lev,d,M,model_pde,U_ptr,F_ptr,DF_ptr,xi); 
}    

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::numflux(int lev,int d,int M, int N,
                                        amrex::Vector<amrex::MultiFab>* U_ptr_m, 
                                        amrex::Vector<amrex::MultiFab>* U_ptr_p,
                                        amrex::Vector<amrex::MultiFab>* F_ptr_m,
                                        amrex::Vector<amrex::MultiFab>* F_ptr_p,
                                        amrex::Vector<amrex::MultiFab>* DF_ptr_m,
                                        amrex::Vector<amrex::MultiFab>* DF_ptr_p)
{
    static_cast<NumericalMethodType*>(this)->numflux(lev,d,M,N,U_ptr_m,U_ptr_p,F_ptr_m,F_ptr_p,DF_ptr_m,DF_ptr_p); 
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::numflux_integral(int lev,int d, int M, int N,
                                                    amrex::Vector<amrex::MultiFab>* U_ptr_m, 
                                                    amrex::Vector<amrex::MultiFab>* U_ptr_p,
                                                    amrex::Vector<amrex::MultiFab>* F_ptr_m,
                                                    amrex::Vector<amrex::MultiFab>* F_ptr_p,
                                                    amrex::Vector<amrex::MultiFab>* DF_ptr_m,
                                                    amrex::Vector<amrex::MultiFab>* DF_ptr_p)
{
    static_cast<NumericalMethodType*>(this)->numflux_integral(lev,d,M,N,U_ptr_m,U_ptr_p,F_ptr_m,
                                                            F_ptr_p,DF_ptr_m,DF_ptr_p); 
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::PlotFile(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                            amrex::Vector<amrex::Vector<amrex::MultiFab>>& X,
                                            int tstep, amrex::Real time) const
{   

    auto varNames = model_pde->getModelVarNames();

    auto _mesh = mesh.lock();
    //Output AMR U_w MFab modal data for all solution components, expected 
    //to then plot the first mode,i.e cell average
    
    //using same timestep for all levels
    amrex::Vector<int> lvl_tstep; 
    for (int l = 0; l <= _mesh->get_finest_lev(); ++l)
    {
        lvl_tstep.push_back(tstep);
    }

    //get number of Mfab components stored in each Mfab of X
    int N = X[0][0].nComp();

    //loop over number of PDEs in the system
    for(int q=0; q<Q; ++q){
        amrex::Vector<std::string> plot_var_name;
        //For the selected PDE solution component,
        //get its name and for eahc of the MFab components
        //add a idx component name m to the string
        //usefull in case we have multiple modes
        for(int m =0 ; m<N; ++m){
            for (size_t i = 0; i < varNames.names.size(); ++i) {
                if (i == q) { 
                    const auto& var = varNames.names[i];    
                    plot_var_name.push_back(var + "_" + std::to_string(m));
                }
            }
        }

        std::string name  = "../Results/Simulation Data/"+out_name_prefix+"_"+std::to_string(tstep)+"_q_"+std::to_string(q)+"_plt";
        const std::string& pltfile_name = name;//amrex::Concatenate(name,5);
        
        //mf to output
        amrex::Vector<const MultiFab*> mf_out;
        for (int l = 0; l <=  _mesh->get_finest_lev(); ++l)
        {
            mf_out.push_back(&X[l][q]);           
        }

        //amrex::WriteSingleLevelPlotfile(pltfile, U_w[q],plot_modes_name, domain_geom, time, 0);
        amrex::WriteMultiLevelPlotfile(pltfile_name, _mesh->get_finest_lev()+1, mf_out, plot_var_name,
                                _mesh->get_Geom(), time, lvl_tstep, _mesh->get_refRatio());
    }
    
} 

#endif 
