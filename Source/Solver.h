#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <memory>

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

//Framework for FDM,FVM,DG and other spectral methods
//for FDM,FVM U_w==U, and thus Np == 1
//numerical methods user defiend classes
//provide both spatial and temporal integrators
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
        void init(std::shared_ptr<EquationType> model_pde)
        {
            Q = model_pde->Q_model;
            Q_unique =model_pde->Q_model_unique;
            
            static_cast<NumericalMethodType*>(this)->init();
        }

        //execute simulation (time-stepping and possible AMR operations)
        void evolve();

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
        void set_initial_condition();

        //declare and init data structures holding system equation 
        void set_init_data_system();

        //declare and init data structures holding single equation 
        void set_init_data_component();

        void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
            ofs = _ofs;
        }

        //General class for numerical methods that use basis decomposition of the solution
        //can maange spatial,temporal and mixed basis functions
        class Basis{
            public:
                Basis() = default;

                ~Basis() = default;

                //Spatial basis function, evaluated at x
                //NB: dim(x) = AMREX_SPACEDIM
                amrex::Real phi_s(int idx, amrex::Vector<amrex::Real> x) const;

                //Spatial basis function first derivative dphi/dx_d, evaluated at x
                amrex::Real dphi_s(int idx, amrex::Vector<amrex::Real> x, int d) const;

                //Spatial basis function second derivative d^2phi/dx_d1dx_d2, evaluated at x
                amrex::Real ddphi_s(int idx, amrex::Vector<amrex::Real> x, int d1, int d2) const;

                //Temporal basis function, evaluated at t
                //NB: dim(t) = 1
                amrex::Real phi_t(int tidx, amrex::Real tau) const;

                //Temporal basis function first derivative dtphi/dt, evaluated at t
                amrex::Real dtphi_t(int tidx, amrex::Real tau) const;

                //Spatio-temporal basis function, evaluated at x
                //NB: dim(x) = AMREX_SPACEDIM+1
                amrex::Real phi_st(int idx, amrex::Vector<amrex::Real> x) const;

                //Set number of basis function/weights/modes Np,mNp
                void set_number_basis();

                //Set spatial basis functions Phi(x) index mapping
                void set_idx_mapping_s();

                //Set temporal basis function Phi(t) index mapping
                void set_idx_mapping_t();

                //Set spatio-temporal basis functions Phi(x,t) index mapping
                void set_idx_mapping_st();

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


        };

        class Quadrature{
            public:
                Quadrature() = default;

                ~Quadrature() = default;

                //Set number of quadrature points (should be func of order p)
                void set_number_quadpoints();

                //Generate the quadrature points
                void set_quadpoints();

                //Vandermonde matrix for mapping modes<->quadrature points
                void set_vandermat();

                void set_inv_vandermat();

                //Vandermonde matrix
                amrex::Vector<amrex::Vector<amrex::Real>> V;
                //  inverse
                amrex::Vector<amrex::Vector<amrex::Real>> Vinv;   
                
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
        };
        

    protected:

        std::shared_ptr<std::ofstream> ofs;

        //number of equations of the system
        int Q; 

        //number of lin-indep equations of the system which are non
        //derivable from others, i.e number of solution unknowns
        //which are independent/unique/not function of others (e.g 
        //not the angular momentum)
        int Q_unique; 

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

};

#endif 
