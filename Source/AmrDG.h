#ifndef AMRDG_H
#define AMRDG_H

#include <string>
#include <limits>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <AMReX_Interpolater.H>

#include "Solver.h"
#include "Mesh.h"

using namespace amrex;

class AmrDG : public Solver<AmrDG>, public std::enable_shared_from_this<AmrDG>
{
  public:
    AmrDG()  = default;

    ~AmrDG();

    void settings(int _p, amrex::Real _T) {
      p = _p;
      T = _T;
    }

    void init();

    void set_init_data_system(int lev,const BoxArray& ba,
                              const DistributionMapping& dm);
    
    class BasisLegendre : public Basis
    {
      public:
        BasisLegendre() = default;

        ~BasisLegendre() = default;

        void init();
        

        void set_number_basis() override;

        void set_idx_mapping_s() override;

        void set_idx_mapping_st() override;

        amrex::Real phi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map, 
                            const amrex::Vector<amrex::Real>& x) const override;
        
        amrex::Real dphi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                            const amrex::Vector<amrex::Real>& x, int d) const override;

        amrex::Real ddphi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
          const amrex::Vector<amrex::Real>& x, int d1, int d2) const override;

        amrex::Real phi_t(int tidx, amrex::Real tau) const override;

        amrex::Real dtphi_t(int tidx, amrex::Real tau) const override;

        amrex::Real phi_st(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                            const amrex::Vector<amrex::Real>& x) const override;

      private:
        int factorial(int n) const;

        amrex::Vector<int> basis_idx_linear; //used for limiting
    };

    class QuadratureGaussLegendre : public Quadrature
    {
      public:
        QuadratureGaussLegendre() = default;

        ~QuadratureGaussLegendre() = default;

        void set_number_quadpoints() override; 

        void set_quadpoints() override;

        void NewtonRhapson(amrex::Real &x, int n); 
    };
      
  private:

    //Vandermonde matrix for mapping modes<->quadrature points
    void set_vandermat();

    void set_inv_vandermat();

    //Element Matrix and Quadrature Matrix
    void set_ref_element_matrix();

    amrex::Real refMat_phiphi(int i,int j, bool is_predictor, bool is_mixed_nmodes) const;

    amrex::Real refBDMat_phiphi(int i,int j, int dim, int xi_bd) const;
    
    amrex::Real refMat_phiDphi(int i,int j, int dim) const;   
    
    amrex::Real refMat_tphitphi(int i,int j) const;
    
    amrex::Real refMat_tphiDtphi(int i,int j) const;
      
    amrex::Real coefficient_c(int k,int l) const;     

    int kroneckerDelta(int a, int b) const;

    
    //Vandermonde matrix
    amrex::Vector<amrex::Vector<amrex::Real>> V;
    //  inverse
    amrex::Vector<amrex::Vector<amrex::Real>> Vinv;   

    //L2 projection quadrature matrix
    amrex::Vector<amrex::Vector<amrex::Real>> quadmat;

    //Mass element matrix for ADER-DG corrector
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_corr;
    
    //Stiffness element matrix for ADER-DG corrector
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Sk_corr;

    //Mass boundary element matrix for ADER-DG corrector and predictor
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Mkbd;

    //Mass element matrix for source term (corrector step)
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_corr_src;

    //Mass element matrix for ADER predictor
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_h_w;
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_h_w_inv;
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_pred;

    //Stiffness element matrix for ADER predictor
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Sk_pred;
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Sk_predVinv;

    //Mass element matrix for source term (predictor step)
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_pred_src;   
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_pred_srcVinv; 

    //ADER predictor vector U(x,t) 
    amrex::Vector<amrex::Vector<amrex::MultiFab>> H;

    //ADER Modal/Nodal predictor vector H_w
    amrex::Vector<amrex::Vector<amrex::MultiFab>> H_w;

    //ADER predictor vector U(x,t) evaluated at boundary plus (+) b+
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> H_p;

    //ADER predictor vector U(x,t) evaluated at boundary minus (-) b-
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> H_m;

    //TODO: mybe nested functions ptr dont need to be shared
    //      also mabye can use again CRTP and define them genrally inside Solver

    std::shared_ptr<BasisLegendre> basefunc;  //TODO:maybe doe snot need to be shared

    std::shared_ptr<QuadratureGaussLegendre> quadrule;  //TODO:maybe doe snot need to be shared
};

///////////////////////////////////////////////////////////////////////////
/*
void AmrDG::set_init_data_system(int lev,const BoxArray& ba,
                                  const DistributionMapping& dm)
{
  H_w[lev].resize(Q); 
  H[lev].resize(Q);  
  H_p[lev].resize(AMREX_SPACEDIM);
  H_m[lev].resize(AMREX_SPACEDIM);

  for(int d=0; d<AMREX_SPACEDIM; ++d){
    H_p[lev][d].resize(Q);
    H_m[lev][d].resize(Q);
  } 
}
*/

/*
class AmrDG : public amrex::AmrCore, public NumericalMethod
{
  public: 

    AmrDG(const RealBox& _rb, int _max_level,const Vector<int>& _n_cell, int _coord, 
          Vector<IntVect> const& _ref_ratios, Array<int,AMREX_SPACEDIM> const& _is_per,
          amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_lo,
          amrex::Vector<amrex::Array<int,AMREX_SPACEDIM>> _bc_hi, 
          amrex::Vector<amrex::Vector<int>> _bc_lo_type,
          amrex::Vector<amrex::Vector<int>> _bc_hi_type, amrex::Real _T,
          amrex::Real _CFL, int _p,
          int _t_regrid, int _t_outplt//, 
          //std::string _limiter_type, amrex::Real _TVB_M, 
          //amrex::Vector<amrex::Real> _AMR_TVB_C,
          //amrex::Vector<amrex::Real> _AMR_curl_C, 
          //amrex::Vector<amrex::Real> _AMR_div_C, 
          //amrex::Vector<amrex::Real> _AMR_grad_C, 
          //amrex::Vector<amrex::Real> _AMR_sec_der_C,
          //amrex::Real _AMR_sec_der_indicator, amrex::Vector<amrex::Real> _AMR_C
          , int _t_limit);

///////////////////////////////////////////////////////////////////////////
    //DG 
    //Initial Conditions, and level initialization
    void InitialCondition(int lev);
    
    amrex::Real Initial_Condition_U_w(int lev,int q,int n,int i,int j,int k) const;
    
    amrex::Real Initial_Condition_U(int lev,int q,int i,int j,int k,
                                    amrex::Vector<amrex::Real> xi) const;
 
      //Modal expansion
    void get_U_from_U_w(int c, amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                        amrex::Vector<amrex::MultiFab>* U_ptr,
                        amrex::Vector<amrex::Real> xi, bool is_predictor);
                                                          
    void get_u_from_u_w(int c, int i, int j, int k,
                        amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                        amrex::Vector<amrex::Array4< amrex::Real>>* u ,
                        amrex::Vector<amrex::Real> xi); 
                        
    void get_u_from_u_w(int c, int i, int j, int k,
                        amrex::Vector<amrex::Array4< amrex::Real>>* uw, 
                        amrex::Vector<amrex::Array4< amrex::Real>>* u ,
                        amrex::Vector<amrex::Real> xi);      
  
                                    
///////////////////////////////////////////////////////////////////////////
//BOUNDARY CODNITION

              
    amrex::Vector<amrex::Vector<amrex::Real>> gDbc_lo;
    amrex::Vector<amrex::Vector<amrex::Real>> gDbc_hi;

    amrex::Vector<amrex::Vector<amrex::Real>> gNbc_lo;
    amrex::Vector<amrex::Vector<amrex::Real>> gNbc_hi;


      
    //Boundary Conditions
    void FillBoundaryCells(amrex::Vector<amrex::MultiFab>* U_ptr, int lev);
    
    amrex::Real gDirichlet_bc(int d, int side, int q) const;
    
    amrex::Real gNeumann_bc(int d, int side, int q) const;

//Boundary Conditions
    amrex::Vector<amrex::Vector<amrex::BCRec>> bc_w; 
    amrex::Vector<amrex::Vector<int>> bc_lo_type;
    amrex::Vector<amrex::Vector<int>> bc_hi_type;
     
    class BoundaryCondition
    {
      public: 
        BoundaryCondition(AmrDG* _amrdg, int _q, int _lev);
                        
        virtual ~BoundaryCondition();
        
        void operator() (const IntVect& iv, Array4<Real> const& dest,
                        const int dcomp, const int numcomp,
                        GeometryData const& geom, const Real time,
                        const BCRec* bcr, const int bcomp,
                        const int orig_comp) const;
                       
      private:
        int boundary_lo_type[AMREX_SPACEDIM];
        int boundary_hi_type[AMREX_SPACEDIM];
        AmrDG* amrdg;
        int q;
        int lev;
    };
///////////////////////////////////////////////////////////////////////////
//LIMTIING/TAGGING



    
    amrex::Real minmodB(amrex::Real a1,amrex::Real a2,amrex::Real a3, 
                        bool &troubled_flag, int l) const;
    
    amrex::Real minmod(amrex::Real a1,amrex::Real a2,amrex::Real a3, 
                        bool &troubled_flag) const;  


    void Limiter_w(int lev); //, amrex::TagBoxArray& tags, char tagval
    
    void Limiter_linear_tvb(int i, int j, int k, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* uw, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* um,
                              amrex::Vector<amrex::Array4<amrex::Real>>* up, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* vw,
                              //amrex::Vector<amrex::Array4<amrex::Real>>* um_cpy,
                              int lev);  

    void AMRIndicator_tvb(int i, int j, int k,
                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                          amrex::Vector<amrex::Array4<amrex::Real>>* um,
                          amrex::Vector<amrex::Array4<amrex::Real>>* up,
                          int l, amrex::Array4<char> const& tag,char tagval, 
                          bool& any_trouble); 
                          
    void AMRIndicator_second_derivative(int i, int j, int k, 
                                        amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                                        int l,amrex::Array4<char> const& tag,
                                        char tagval, bool& any_trouble);    
                                        
    void AMRIndicator_curl(int i, int j, int k,
                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                          amrex::Array4<amrex::Real> const & curl_indicator,int l, 
                          bool flag_local,amrex::Array4<char> const& tag,
                          char tagval,bool& any_trouble);
                          
    void AMRIndicator_div(int i, int j, int k,
                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw, 
                          amrex::Array4<amrex::Real> const & div_indicator,int l, 
                          bool flag_local,amrex::Array4<char> const& tag,
                          char tagval,bool& any_trouble);
                          
    void AMRIndicator_grad(int i, int j, int k,
                          amrex::Vector<amrex::Array4<const  amrex::Real>>* uw, 
                          amrex::Array4<amrex::Real> const & grad_indicator,int l, 
                          bool flag_local,amrex::Array4<char> const& tag,
                          char tagval,bool& any_trouble);


    
    //Model Equation/Simulation settings and variables    
    
    int t_limit;
    std::string limiter_type;


    amrex::Vector<amrex::Real> AMR_C;
    
    amrex::Real AMR_curl_indicator;
    amrex::Vector<amrex::Real> AMR_curl_C;
    amrex::Vector<amrex::MultiFab> idc_curl_K;

    amrex::Real AMR_div_indicator;
    amrex::Vector<amrex::Real> AMR_div_C;
    amrex::Vector<amrex::MultiFab> idc_div_K;
    
    amrex::Real AMR_grad_indicator;
    amrex::Vector<amrex::Real> AMR_grad_C;
    amrex::Vector<amrex::MultiFab> idc_grad_K;    
    
    amrex::Real AMR_sec_der_indicator;
    amrex::Vector<amrex::Real> AMR_sec_der_C;
    
    amrex::Real TVB_M;
    amrex::Vector<amrex::Real> AMR_TVB_C;
///////////////////////////////////////////////////////////////////////////


  
  private: 


   

    
    //ADER-DG    
    void ADER();
    
    void ComputeDt();
    
    void Predictor_set(const amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                      amrex::Vector<amrex::MultiFab>* H_w_ptr);
                      
    void Update_H_w(int lev, int q);
    
    void Update_U_w(int lev, int q);
    
  
    
    //Non-linear fluxes, Numerical fluxes
                                                
    void Flux(int lev,int d, int M, 
              amrex::Vector<amrex::MultiFab>* U_w_ptr, 
              amrex::Vector<amrex::MultiFab>* U_ptr,
              amrex::Vector<amrex::MultiFab>* F_ptr,
              amrex::Vector<amrex::MultiFab>* DF_ptr,
              amrex::Vector<amrex::Vector<amrex::Real>> xi,
              bool flag_bd, bool is_predictor);
              
    void Source(int lev,int M,
                amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                amrex::Vector<amrex::MultiFab>* U_ptr,
                amrex::Vector<amrex::MultiFab>* S_ptr,
                amrex::Vector<amrex::Vector<amrex::Real>> xi, 
                bool is_predictor);
              
    void InterfaceNumFlux(int lev,int d,int M, 
                          amrex::Vector<amrex::MultiFab>* U_ptr_m, 
                          amrex::Vector<amrex::MultiFab>* U_ptr_p);
    
    void InterfaceNumFlux_integrate(int lev,int d,int M);
                                      
    amrex::Real NumericalFlux(int d, int m,int i, int j, int k, 
                              amrex::Array4<const amrex::Real> up, 
                              amrex::Array4<const amrex::Real> um, 
                              amrex::Array4<const amrex::Real> fp,
                              amrex::Array4<const amrex::Real> fm,  
                              amrex::Array4<const amrex::Real> dfp,
                              amrex::Array4<const amrex::Real> dfm);
                              
    amrex::Real PhysicalFlux(int lev, int d,int q, int m, int i, int j, int k,
                            amrex::Vector<amrex::Array4<const amrex::Real>>* u,                             
                            amrex::Vector<amrex::Real> xi);
                            
    amrex::Real DPhysicalFlux(int lev, int d,int q, int m, int i, int j, int k,
                            amrex::Vector<amrex::Array4<const amrex::Real>>* u,                             
                            amrex::Vector<amrex::Real> xi);
                              
    amrex::Real PhysicalSource(int lev,int q, int m, int i, int j, int k,
                              amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                              amrex::Vector<amrex::Real> xi);

    //I/O and MISC
    void NormDG();//TODO:
    
    void PlotFile(int tstep, amrex::Real time) const;
    
    void L1Norm_DG_AMR();

    void L2Norm_DG_AMR();   
    
    void LpNorm_DG_AMR(int _p, amrex::Vector<amrex::Vector<amrex::Real>> quad_pt, int N) const;
    
    void DEBUG_print_MFab();
    
    void Conservation(int lev, int M, amrex::Vector<amrex::Vector<amrex::Real>> xi, int d);
};
   
extern AMREX_EXPORT AmrDG::DGprojInterp custom_interp;
*/
#endif

/*------------------------------------------------------------------------*/
/*
VARIABLES NAMES NOTATION
q   :   variable to iterate across solution components U=[u1,...,uq,...,uQ], it 
        is used also for quadrature loops to indicate q-th quadrature point
d   :   variable to iterate across dimensions
_w  :   indicates that the data is modal
p   :   positive/plus/+, also indicates the order of DG scheme
m   :   negative/minus/- ,
bd  :   data has been evaluated at boundary location
num :   numerical
c   :   MultiFab component indexing 
n   :   used to iterate until Np
m   :   used to iterate until Mp
l   :   used to iterate until L (levels)
x   :   used to represent point in domain
xi  :   used to represent point in reference domain
OBSERVATIONS
-MFiter are done differently depending on if we use MPI or MPI+OpenMP
  if MPI: use static tiling, no parallelizatition of tile operations
  if MPI+OMP: use dynamic tiling, each tile is given to a thread and then also 
  the mesh loop is parallelized between the threads
-some functions require to pass a pointer to either U_w or H_w, this is done because
 in this way it is easier to e.g use the same functions in the context of Runge-Kutta
*/
/*------------------------------------------------------------------------*/