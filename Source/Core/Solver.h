#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <memory>
#include <functional>

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

        //Set validation mode: if true, use analytical IC at all levels (for convergence tests)
        //If false (default), levels > 0 use projection from coarser level
        void setValidationMode(bool use_analytical_ic) {
            flag_analytical_ic = use_analytical_ic;
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
        void get_U_from_U_w(int M, int N,amrex::Vector<amrex::MultiFab>& _U,
                            amrex::Vector<amrex::MultiFab>& _U_w,
                            const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

        template <typename EquationType>
        void source(int lev,int M,
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::Vector<amrex::MultiFab>& _U,
                    amrex::Vector<amrex::MultiFab>& _S,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

        template <typename EquationType>
        void flux(int lev,int d, int M,
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::Vector<amrex::MultiFab>& _U,
                    amrex::Vector<amrex::MultiFab>& _F,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

        template <typename EquationType>
        void flux_bd(int lev,int d, int M,
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::Vector<amrex::MultiFab>& _U,
                    amrex::Vector<amrex::MultiFab>& _F,
                    amrex::Vector<amrex::MultiFab>& _DF,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

        void numflux(int lev,int d,int M, int N,
                    amrex::Vector<amrex::MultiFab>& _U_m,
                    amrex::Vector<amrex::MultiFab>& _U_p,
                    amrex::Vector<amrex::MultiFab>& _F_m,
                    amrex::Vector<amrex::MultiFab>& _F_p,
                    amrex::Vector<amrex::MultiFab>& _DF_m,
                    amrex::Vector<amrex::MultiFab>& _DF_p);

        //computes the integral of the numerical flux across interface
        void numflux_integral(int lev,int d,int M, int N,
                            amrex::Vector<amrex::MultiFab>& _U_m,
                            amrex::Vector<amrex::MultiFab>& _U_p,
                            amrex::Vector<amrex::MultiFab>& _F_m,
                            amrex::Vector<amrex::MultiFab>& _F_p,
                            amrex::Vector<amrex::MultiFab>& _DF_m,
                            amrex::Vector<amrex::MultiFab>& _DF_p);

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

        void AMR_sync_initial_condition();

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
                                                    int tstep, amrex::Real time, int level = -1) const;
                                                
        //Apply boundary conditions by calling BC methods
        template <typename EquationType>
        void FillBoundaryCells(const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                                amrex::Vector<amrex::MultiFab>& _U,
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

                //number of quadrature points for quadrature of surface (spatio) integral
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

        //Internal bridge: connects non-template ErrorEst to template model_pde->pde_tag_cell_refinement
        std::function<bool(int, int, int, int, amrex::Real, amrex::Real)> m_tag_impl;

        //number of equations of the system
        int Q; 

        //number of lin-indep equations of the system which are non
        //derivable from others, i.e number of solution unknowns
        //which are independent/unique/not function of others (e.g 
        //not the angular momentum)
        int Q_unique; 

        //Flag to indicate if source term is considered
        bool flag_source_term;

        //Flag to indicate if analytical IC should be used at all levels (for validation/convergence tests)
        //If false, levels > 0 use projection from coarser level
        bool flag_analytical_ic = false;

        //spatial (approxiamtion) order
        int p;

        //Courant–Friedrichs–Lewy number
        amrex::Real CFL;

        //Safety factor for time-step size computation
        amrex::Real c_dt;

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

        //Physical flux F(x,t) evaluated at boundary minus (-) b-
        amrex::Vector<amrex::Vector<amrex::MultiFab>> Fm;
        //  derivative
        amrex::Vector<amrex::Vector<amrex::MultiFab>> DFm;

        //Physical flux F(x,t) evaluated at boundary plus (+) b+
        amrex::Vector<amrex::Vector<amrex::MultiFab>> Fp;
        //  derivative
        amrex::Vector<amrex::Vector<amrex::MultiFab>> DFp;

        //Numerical flux approximation Fnum(x,t)
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> Fnum;

        //Numerical flux approximation Fnum(x,t) integrated at fine lvl
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> Fnum_int_f;

        //Numerical flux approximation Fnum(x,t) integrated at coarse lvl
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> Fnum_int_c;

        //Store at each level the mask to identify cells
        //at coarse-fine interface (0 = uncovered, 1 = covered by finer level)
        amrex::Vector<amrex::iMultiFab> coarse_fine_interface_mask;

        //Valid-cell mask for each level: 1 = valid cell on this level, 0 = not.
        //Ghost cells filled via FillBoundary (handles periodicity correctly).
        //Used for periodic-aware fine-coarse interface detection in numflux.
        amrex::Vector<amrex::iMultiFab> fine_level_valid_mask;

        //Precomputed C-F interface face data per level per dimension [lev][d]
        //Face-centered iMultiFabs, 1 component, 0 ghost cells
        //b_coarse: +1/-1 = interface direction (with ownership rule), 0 = not C-F
        amrex::Vector<amrex::Vector<amrex::iMultiFab>> cf_face_b_coarse;
        //b_fine: +1/-1 = interface direction, 0 = not F-C
        amrex::Vector<amrex::Vector<amrex::iMultiFab>> cf_face_b_fine;
        //child sub-face index (0 to 2^(D-1)-1), valid only where cf_face_b_fine != 0
        amrex::Vector<amrex::Vector<amrex::iMultiFab>> cf_face_child_idx;

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
        // Hide cursor during progress bar display
        std::cout << "\033[?25l" << std::flush;

        // Bar width 50 + brackets/text ~30 = ~80 chars (classic terminal width)
        m_bar.emplace(
            indicators::option::BarWidth{100},
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
        std::cout << std::flush;
    }

    //Set up tagging bridge: connects non-template ErrorEst to model_pde->pde_tag_cell_refinement
    m_tag_impl = [model_pde, _mesh](int lev, int i, int j, int k,
                                     amrex::Real time, amrex::Real amr_c_lev) -> bool {
      return model_pde->pde_tag_cell_refinement(lev, i, j, k, time, amr_c_lev, _mesh);
    };

    //AmrCore.h function initialize multilevel mesh, geometry, Box array and DistributionMap
    //calls MakeNewLevelFromScratch
    //Initialize BoxArray, DistributionMapping and data from scratch.
    //Calling this function requires the derive class implement its own MakeNewLevelFromScratch
    //to allocate and initialize data.

    //NB: Construct full domain grids up to max_level
    _mesh->InitFromScratch(time);
    // At this point: all levels 0..max_level exist as full-domain grids (AMReX's InitFromScratch creates uniform grids at every level). All data is
    // zero. All possible levels have been created

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

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::set_initial_condition(const std::shared_ptr<ModelEquation<EquationType>>& model_pde)
{
    auto _mesh = mesh.lock();

    //Define IC on single coarse mesh
    //analytical IC on level 0
    static_cast<NumericalMethodType*>(this)->set_initial_condition(model_pde,0);

    if (_mesh->L > 1) {
        for(int l=1; l<_mesh->L; ++l){
            if (flag_analytical_ic) {
                //Validation mode: use analytical IC at all levels
                static_cast<NumericalMethodType*>(this)->set_initial_condition(model_pde,l);
            } else {
                //Normal AMR mode: project from coarser level
                AMR_interpolate_initial_condition(l);
            }
        }

        // Sync all levels after IC population:
        // - Analytical IC: average fine→coarse for conservation, then FillPatch all ghost cells
        // - Projection IC: FillPatch syncs same-level + fine-coarse ghost cells
        AMR_sync_initial_condition();
    }
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_interpolate_initial_condition(int lev)
{
    static_cast<NumericalMethodType*>(this)->AMR_interpolate_initial_condition(lev);
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::AMR_sync_initial_condition()
{
    static_cast<NumericalMethodType*>(this)->AMR_sync_initial_condition();
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
                                                    amrex::Vector<amrex::MultiFab>& _U,
                                                    int lev, amrex::Real time)

{
    //static_cast<NumericalMethodType*>(this)->FillBoundaryCells(mesh,bdcond,_U,lev,time);
    //TODO: in case AmrDG should prvide some specialized functionalities to BC maybe then static_cast is needed
    bdcond->FillBoundaryCells(_U, lev, time);
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
void Solver<NumericalMethodType>::get_U_from_U_w(int M, int N,amrex::Vector<amrex::MultiFab>& _U,
                                                amrex::Vector<amrex::MultiFab>& _U_w,
                                                const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    static_cast<NumericalMethodType*>(this)->get_U_from_U_w(M,N,_U,_U_w,xi);
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::source(int lev,int M,
                                        const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        amrex::Vector<amrex::MultiFab>& _U,
                                        amrex::Vector<amrex::MultiFab>& _S,
                                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    static_cast<NumericalMethodType*>(this)->source(lev,M,model_pde,_U,_S,xi);
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::flux(int lev,int d, int M,
                                        const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        amrex::Vector<amrex::MultiFab>& _U,
                                        amrex::Vector<amrex::MultiFab>& _F,
                                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    static_cast<NumericalMethodType*>(this)->flux(lev,d,M,model_pde,_U,_F,xi);
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::flux_bd(int lev,int d, int M,
                                        const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        amrex::Vector<amrex::MultiFab>& _U,
                                        amrex::Vector<amrex::MultiFab>& _F,
                                        amrex::Vector<amrex::MultiFab>& _DF,
                                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    static_cast<NumericalMethodType*>(this)->flux_bd(lev,d,M,model_pde,_U,_F,_DF,xi);
}    

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::numflux(int lev,int d,int M, int N,
                                        amrex::Vector<amrex::MultiFab>& _U_m,
                                        amrex::Vector<amrex::MultiFab>& _U_p,
                                        amrex::Vector<amrex::MultiFab>& _F_m,
                                        amrex::Vector<amrex::MultiFab>& _F_p,
                                        amrex::Vector<amrex::MultiFab>& _DF_m,
                                        amrex::Vector<amrex::MultiFab>& _DF_p)
{
    static_cast<NumericalMethodType*>(this)->numflux(lev,d,M,N,_U_m,_U_p,_F_m,_F_p,_DF_m,_DF_p);
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::numflux_integral(int lev,int d, int M, int N,
                                                    amrex::Vector<amrex::MultiFab>& _U_m,
                                                    amrex::Vector<amrex::MultiFab>& _U_p,
                                                    amrex::Vector<amrex::MultiFab>& _F_m,
                                                    amrex::Vector<amrex::MultiFab>& _F_p,
                                                    amrex::Vector<amrex::MultiFab>& _DF_m,
                                                    amrex::Vector<amrex::MultiFab>& _DF_p)
{
    static_cast<NumericalMethodType*>(this)->numflux_integral(lev,d,M,N,_U_m,_U_p,_F_m,
                                                            _F_p,_DF_m,_DF_p);
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::PlotFile(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                            amrex::Vector<amrex::Vector<amrex::MultiFab>>& X,
                                            int tstep, amrex::Real time, 
                                            int level) const 
{   
    auto varNames = model_pde->getModelVarNames();
    auto _mesh = mesh.lock();
    
    // Determine bounds based on the 'level' argument
    int start_lev = (level < 0) ? 0 : level;
    int end_lev   = (level < 0) ? _mesh->get_finest_lev() : level;
    int num_levs  = end_lev - start_lev + 1;

    for(int q = 0; q < Q; ++q) {
        amrex::Vector<std::string> plot_var_name;
        int nComp = X[0][q].nComp();
        for(int m = 0; m < nComp; ++m) {
            plot_var_name.push_back(varNames.names[q] + "_" + std::to_string(m));
        }

        // Keep your original naming convention
        std::string pltfile_name = "../Results/Simulation Data/" + out_name_prefix + "_" + 
                                   std::to_string(tstep) + "_q_" + std::to_string(q) + "_plt";

        // Containers for the writer
        amrex::Vector<const amrex::MultiFab*> mf_out;
        amrex::Vector<amrex::Geometry> geom_out;
        amrex::Vector<int> lvl_tstep;
        amrex::Vector<amrex::IntVect> ref_ratio;

        // Build vectors only for the levels we want
        for (int l = start_lev; l <= end_lev; ++l) {
            mf_out.push_back(&X[l][q]);
            geom_out.push_back(_mesh->get_Geom(l));
            lvl_tstep.push_back(tstep);
            if (l < end_lev) {
                ref_ratio.push_back(_mesh->get_refRatio(l));
            }
        }

        // Use the MultiLevel writer for both cases to keep naming/structure identical
        amrex::WriteMultiLevelPlotfile(pltfile_name, num_levs, mf_out, plot_var_name,
                                       geom_out, time, lvl_tstep, ref_ratio);
    }
}

#endif 
