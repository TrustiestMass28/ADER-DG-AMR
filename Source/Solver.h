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

template <typename NumericalMethodType>
class Mesh;

using namespace amrex;

template <typename NumericalMethodType>
class Solver
{
    public: 

        Solver() = default;

        virtual ~Solver() = default;

        template <typename... Args>
        void settings(Args... args) {
            std::cout << "test" << std::endl;
            static_cast<NumericalMethodType*>(this)->settings(std::forward<Args>(args)...);
        }

        //TODO: pass all tempaltes of other classes from which Solver might need data to init
        //like stuff from geometry for number of levels,...
        template <typename EquationType>
        void init(std::shared_ptr<EquationType> model_pde, std::shared_ptr<Mesh<NumericalMethodType>> _mesh);

        //execute simulation (time-stepping and possible AMR operations)
        template <typename EquationType>
        void evolve(std::shared_ptr<EquationType> model_pde);

        //perform a time-step, advance solution by one time-step
        void time_integration();

        //compute and set time-step size
        void set_Dt();

        //get solution vector evaluation
        void get_U();

        //get solution vector evaluation based on its decomposition
        void get_U_from_U_w();

        //get solution vector derivative
        void get_dU();

        //sometimes IC is for modes/weights, other for actual sol vector
        //depending on num method it will call 
        template <typename EquationType>
        void set_initial_condition(std::shared_ptr<EquationType> model_pde);
        
        //declare and init data structures holding system equation 
        void set_init_data_system(int lev,const BoxArray& ba,
                                    const DistributionMapping& dm);

        //declare and init data structures holding single equation 
        void set_init_data_component(int lev,const BoxArray& ba, 
                                        const DistributionMapping& dm, int q);
        
        void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
            ofs = _ofs;
        }

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

                void setNumericalMethod(NumericalMethodType* _numme);

            protected:
                //Ptr used to access numerical method and solver data
                NumericalMethodType* numme;

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

                void setNumericalMethod(NumericalMethodType* _numme);
                
                protected:
                    //Ptr used to access numerical method and solver data
                    NumericalMethodType* numme;
            };

    protected:
        std::shared_ptr<std::ofstream> ofs;

        std::shared_ptr<Mesh<NumericalMethodType>> mesh;

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
        int dt_outplt;   //data output physical time interval

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

    private:
    
        void setMesh(std::shared_ptr<Mesh<NumericalMethodType>> _mesh);

};

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::init(std::shared_ptr<EquationType> model_pde, std::shared_ptr<Mesh<NumericalMethodType>> _mesh) 
{
    setMesh(_mesh);

    //Get model specific data that influence numerical set-up
    Q = model_pde->Q_model;
    Q_unique = model_pde->Q_model_unique;
    flag_source_term = model_pde->flag_source_term;

    //Numerical method specific initialization
    static_cast<NumericalMethodType*>(this)->init();

    const Real time = 0.0;
    mesh->InitFromScratch(time);    //AmrCore
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::setMesh(std::shared_ptr<Mesh<NumericalMethodType>> _mesh)
{
    mesh = _mesh;
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
void Solver<NumericalMethodType>::set_initial_condition(std::shared_ptr<EquationType> model_pde)
{
    
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::Basis::setNumericalMethod(NumericalMethodType* _numme)
{
    numme = _numme;
}

template <typename NumericalMethodType>
Solver<NumericalMethodType>::Basis::~Basis()
{
    delete numme;
}

template <typename NumericalMethodType>
void Solver<NumericalMethodType>::Quadrature::setNumericalMethod(NumericalMethodType* _numme)
{
    numme = _numme;
}

template <typename NumericalMethodType>
Solver<NumericalMethodType>::Quadrature::~Quadrature()
{
    delete numme;
}

template <typename NumericalMethodType>
template <typename EquationType>
void Solver<NumericalMethodType>::evolve(std::shared_ptr<EquationType> model_pde)
{
    static_cast<NumericalMethodType*>(this)->evolve(model_pde); 
}

#endif 
