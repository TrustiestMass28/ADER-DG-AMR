#ifndef AMRDG_H
#define AMRDG_H

#include <string>
#include <limits>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

//#include "NumericalMethod.h"
#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <AMReX_Interpolater.H>

#include "Solver.h"

using namespace amrex;

class AmrDG : public Solver<AmrDG>
{
  public:
    AmrDG()  = default;

    ~AmrDG() = default;

    void settings(int _p, amrex::Real _T) {
      p = _p;
      T = _T;
    }

    void init(){ std::cout << "HERE" << std::endl;}

    class BasisLegendre : public Basis
    {

    }

};

/*
BASIS FUNCTION STUFF

//Basis Function

  amrex::Real Phi(int idx, amrex::Vector<amrex::Real> x) const;

  amrex::Real DPhi(int idx, amrex::Vector<amrex::Real> x, int d) const;

  amrex::Real DDPhi(int idx, amrex::Vector<amrex::Real> x, int d1, int d2) const;

  amrex::Real Dtphi(int tidx, amrex::Real tau) const;

  amrex::Real tphi(int tidx, amrex::Real tau) const;    

  amrex::Real modPhi(int idx, amrex::Vector<amrex::Real> x) const;

  //Basis Functionclear
  void PhiIdxGenerator_s();

  void PhiIdxGenerator_st();

  void number_modes();

  amrex::Vector<amrex::Vector<int>> mat_idx_s; 
  amrex::Vector<int> lin_mode_idx;

  int mNp;        //number of modes for modified basis function modphi_i


  //Basis function d.o.f
  amrex::Vector<amrex::Vector<int>> mat_idx_st; 
  //used to store the combinations of indices of 1d Legendre polynomials: e.g
  // mat_idx[5] = [0,1,4] ==> phi_5= P_0*P_1*P_4

    

QUADRATURE

  void number_quadintpts();

  void GenerateQuadPts();

  void VandermondeMat();

  void InvVandermondeMat();

  amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_GLquad_L2proj;

  int qMp;        //number of quadrature points for quadrature of volume integral
  int qMp_L2proj; //number of quadrature points only in space, used for the BCs,ICs,
  int qMpbd;      //number of quadrature points for quadrature of surface integral
  int qMp_1d;     //number of quadrature points in 1 dimension, powers of this 
          //leads to qMp and qMpbd

  amrex::Vector<amrex::Vector<amrex::Real>> volquadmat;
  amrex::Vector<amrex::Vector<amrex::Real>> L2proj_quadmat;



  //Interpolation nodes and Gauss-Legendre Quadrature points
  amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_GLquad_s;
  amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_GLquad_t;
  amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_GLquad;
  amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> xi_ref_GLquad_bdm;
  amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> xi_ref_GLquad_bdp;

  amrex::Vector<amrex::Vector<amrex::Real>> xi_ref_equidistant;  
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
          
    AmrDG() = default;

    virtual ~AmrDG();
    
    void Init();
    
    //void Evolve();

 
  
    //AmrCore pure virtual functions, need to provide custom implementation
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, 
                                        const amrex::BoxArray& ba,
                                        const amrex::DistributionMapping& dm) override;  
                                          
    virtual void MakeNewLevelFromCoarse(int lev, amrex::Real time,
                                        const amrex::BoxArray& ba, 
                                        const amrex::DistributionMapping& dm) override;
                                        
    virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba,
                            const amrex::DistributionMapping& dm) override;
                            
    virtual void ErrorEst (int lev, amrex::TagBoxArray& tags, 
                          amrex::Real time, int ngrow) override;
                          
    virtual void ClearLevel (int lev) override;
 
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


    

    
    //Element Matrix and Quadrature Matrix
    
    amrex::Real RefMat_phiphi(int i,int j, bool is_predictor, bool is_mixed_nmodes) const;
    
    amrex::Real RefBDMat_phiphi(int i,int j, int dim, int xi_bd) const;
    
    amrex::Real RefMat_phiDphi(int i,int j, int dim) const;   
    
    amrex::Real RefMat_tphitphi(int i,int j) const;
    
    amrex::Real RefMat_tphiDtphi(int i,int j) const;
      
    amrex::Real Coefficient_c(int k,int l) const;     
 
    

    
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
  
    
    //MISC
    int factorial(int n) const;
        
    int KroneckerDelta(int a, int b) const;
    
    void NewtonRhapson(amrex::Real &x, int n); 
    
    amrex::Real minmodB(amrex::Real a1,amrex::Real a2,amrex::Real a3, 
                        bool &troubled_flag, int l) const;
    
    amrex::Real minmod(amrex::Real a1,amrex::Real a2,amrex::Real a3, 
                        bool &troubled_flag) const;  
                        
              
    amrex::Vector<amrex::Vector<amrex::Real>> gDbc_lo;
    amrex::Vector<amrex::Vector<amrex::Real>> gDbc_hi;

    amrex::Vector<amrex::Vector<amrex::Real>> gNbc_lo;
    amrex::Vector<amrex::Vector<amrex::Real>> gNbc_hi;
    

  
  private: 
  

        
   
    
    //Element Matrix and Quadrature Matrix
    void MatrixGenerator();
    

    

    
    //Initial Conditions, and level initialization
    void InitialCondition(int lev);
    
    amrex::Real Initial_Condition_U_w(int lev,int q,int n,int i,int j,int k) const;
    
    amrex::Real Initial_Condition_U(int lev,int q,int i,int j,int k,
                                    amrex::Vector<amrex::Real> xi) const;
    
   
    
    //Boundary Conditions
    void FillBoundaryCells(amrex::Vector<amrex::MultiFab>* U_ptr, int lev);
    
    amrex::Real gDirichlet_bc(int d, int side, int q) const;
    
    amrex::Real gNeumann_bc(int d, int side, int q) const;


    
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

  
    
    //AMR level creation/destruction/ghost cell sync
    void InitData_system(int lev,const BoxArray& ba, 
                        const DistributionMapping& dm);
    
    void InitData_component(int lev,const BoxArray& ba, 
                            const DistributionMapping& dm, int q);
    
    void FillPatch (int lev, Real time, amrex::MultiFab& mf, int icomp, int ncomp, int q);
    
    void FillCoarsePatch (int lev, Real time, amrex::MultiFab& mf, int icomp, int ncomp, int q);
    
    void GetData (int lev, int q, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime);
    
    void AverageFineToCoarse();    
    
    void AverageFineToCoarseFlux(int lev);
    
    void FillPatchGhostFC(int lev,amrex::Real time,int q);

  
    
    //AMR refinement and limiting
    void AMR_settings_tune();
    
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

    //DG settings

    int nghost= 1;    
 

    
    //AMR settings 
    int L;
    int t_regrid;  

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

    
    //Multifabs vectors (LxDxQ or LxQ)
    //L:  max number of levels
    //D:  dimensions
    //Q:  number of solution components
    
    //Nodal and Modal MFs containers   
    amrex::Vector<amrex::Vector<amrex::MultiFab>> H_w;
    amrex::Vector<amrex::Vector<amrex::MultiFab>> H;
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> H_p;
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::MultiFab>>> H_m;
        


    //Element (analytical) Matrices and Quadrature matrices
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_corr;
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Sk_corr;
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Mkbd;
    
    amrex::Vector<amrex::Vector<amrex::Real>> V;
    amrex::Vector<amrex::Vector<amrex::Real>> Vinv;   
    
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_h_w;
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_h_w_inv;
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_pred;
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Sk_pred;
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Sk_predVinv;
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_s;   
    amrex::Vector<amrex::Vector<amrex::Real>> Mk_sVinv;


   

  
    
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