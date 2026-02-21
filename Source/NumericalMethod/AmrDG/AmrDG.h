#ifndef AMRDG_H
#define AMRDG_H

#include <string>
#include <limits>
#include <numeric>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <AMReX_Interpolater.H>
#include <AMReX_BoxDomain.H>
#include <Eigen/Core>


#include "Solver.h"
#include "Mesh.h"

using namespace amrex;

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

class AmrDG : public Solver<AmrDG>, public std::enable_shared_from_this<AmrDG>
{
  public:
    //type alias to improve readibility
    using NumericalMethodType = AmrDG;

    AmrDG()  = default;

    ~AmrDG();

    void settings(int _p, amrex::Real _T, amrex::Real _c_dt);

    void init();

    void init_bc(amrex::Vector<amrex::Vector<amrex::BCRec>>& bc, int& n_comp);

    template <typename EquationType>
    void evolve(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                const std::shared_ptr<BoundaryCondition<EquationType,
                NumericalMethodType>>& bdcond);

    template <typename EquationType>
    void time_integration(  const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                            const std::shared_ptr<BoundaryCondition<EquationType,
                            NumericalMethodType>>& bdcond,
                            amrex::Real time);

    template <typename EquationType>
    void ADER(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
              const std::shared_ptr<BoundaryCondition<EquationType,
              NumericalMethodType>>& bdcond,
              amrex::Real time);

    template <typename EquationType>
    void set_Dt(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);

    //void AMR_avg_down_initial_condition();

    void AMR_advanced_settings();

    void AMR_interpolate_initial_condition(int lev);

    void AMR_sync_initial_condition();

    void AMR_average_fine_coarse();

    void AMR_clear_level_data(int lev);

    void AMR_tag_cell_refinement(int lev, amrex::TagBoxArray& tags, 
                                amrex::Real time, int ngrow);

    void AMR_remake_level(int lev, amrex::Real time, const amrex::BoxArray& ba,
                          const amrex::DistributionMapping& dm);

    void AMR_make_new_fine_level(int lev, amrex::Real time,
                                const amrex::BoxArray& ba, 
                                const amrex::DistributionMapping& dm);

    void AMR_FillFromCoarsePatch (int lev, Real time, amrex::MultiFab* fmf,
                              int icomp,int ncomp);

    void AMR_FillPatch(int lev, Real time, amrex::MultiFab* mf,
                      int icomp, int ncomp);

    void AMR_set_flux_registers();

    void AMR_flux_correction();

    void set_init_data_system(int lev,const BoxArray& ba,
                              const DistributionMapping& dm);

    void set_init_data_component(int lev,const BoxArray& ba,
                                const DistributionMapping& dm, int q);

    template <typename EquationType> 
    void set_initial_condition(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, int lev);

    template <typename EquationType> 
    amrex::Real set_initial_condition_U_w(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                      int lev,int q,int n,int i,int j,int k);

    template <typename EquationType> 
    amrex::Real set_initial_condition_U(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        int lev,int q,int i,int j,int k,
                                        const amrex::Vector<amrex::Real>& xi) const;

    void get_U_from_U_w(int M, int N, amrex::MultiFab* _U,
                        amrex::MultiFab* _U_w,
                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    template <typename EquationType>
    void source(int lev, int M,
                const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                amrex::MultiFab* _U,
                amrex::MultiFab* _S,
                const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    template <typename EquationType>
    void flux(int lev, int d, int M,
              const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
              amrex::MultiFab* _U,
              amrex::MultiFab* _F,
              const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    template <typename EquationType>
    void flux_bd(int lev, int d, int M,
                 const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                 amrex::MultiFab* _U,
                 amrex::MultiFab* _F,
                 amrex::MultiFab* _DF,
                 const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    void numflux(int lev,int d,int M, int N,
                 amrex::MultiFab* _U_m,
                 amrex::MultiFab* _U_p,
                 amrex::MultiFab* _F_m,
                 amrex::MultiFab* _F_p,
                 amrex::MultiFab* _DF_m,
                 amrex::MultiFab* _DF_p);

    amrex::Real LLF_numflux(int d, int m,int i, int j, int k, 
                amrex::Array4<const amrex::Real> up, 
                amrex::Array4<const amrex::Real> um, 
                amrex::Array4<const amrex::Real> fp,
                amrex::Array4<const amrex::Real> fm,  
                amrex::Array4<const amrex::Real> dfp,
                amrex::Array4<const amrex::Real> dfm);

    amrex::Real setBC(const amrex::Vector<amrex::Real>& bc, int comp,int dcomp,
                      int q, int lev);
    
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

    
    class L2ProjInterp : public AMR_Interpolation<L2ProjInterp>
    {
      public:
        L2ProjInterp() = default;

        ~L2ProjInterp() = default;

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
                    RunOn            runon);

        void amr_scatter(int i, int j, int k, Array4<Real> const& fine, 
                                            int fcomp, Array4<Real const> const& crse, 
                                            int ccomp, int ncomp, IntVect const& ratio) noexcept;
                                            
        void average_down(const MultiFab& S_fine, int fine_comp, MultiFab& S_crse, 
                          int crse_comp, int ncomp, const IntVect& ratio, 
                          const int lev_fine, const int lev_coarse) noexcept;
        
        //AMR gater MUST be called from average down
        void amr_gather(int i, int j, int k,  Array4<Real const> const& fine,int fcomp,
                        Array4<Real> const& crse, int ccomp, 
                        int ncomp, IntVect const& ratio ) noexcept;

        const Eigen::MatrixXd& get_flux_proj_mat(int d, int child_idx, int b) const ;

        void reflux(amrex::MultiFab& U_crse, const amrex::MultiFab& correction_mf,
                    int lev, const amrex::Geometry& crse_geom) noexcept;

        Box CoarseBox (const Box& fine, int ratio);

        Box CoarseBox (const Box& fine, const IntVect& ratio);

        void interp_proj_mat();

        void flux_proj_mat();
        
      private:      

        struct IndexMap{
          int i;
          int j;
          int k;
          int fidx;
        };

        //pass fine cell index and return overlapping coarse cell index 
        //and index locating fine cell w.r.t coarse one reference frame
        IndexMap set_fine_coarse_idx_map(int i, int j, int k, const amrex::IntVect& ratio);

        //pass coarse cell index and return all fine cells indices and their
        //respective rf-element indices to lcoate them w.r.t coarse cell
        amrex::Vector<IndexMap> set_coarse_fine_idx_map(int i, int j, int k, const amrex::IntVect& ratio);

        amrex::Vector<Eigen::MatrixXd> P_cf;

        amrex::Vector<Eigen::MatrixXd> P_fc;      

        // Matrices for projecting fine fluxes onto the COARSE LOW interface (xi = -1)
        amrex::Vector<amrex::Vector<Eigen::MatrixXd>> P_flux_fc_low;

        // Matrices for projecting fine fluxes onto the COARSE HIGH interface (xi = +1)
        amrex::Vector<amrex::Vector<Eigen::MatrixXd>> P_flux_fc_high;

        Eigen::MatrixXd M;

        Eigen::MatrixXd Minv;

    };

  protected:

    int kroneckerDelta(int a, int b) const;

    amrex::Real refMat_phiphi(int j, const amrex::Vector<amrex::Vector<int>>& idx_map_j, 
                              int i, const amrex::Vector<amrex::Vector<int>>& idx_map_i) const ;

    //L2 projection quadrature matrix
    Eigen::MatrixXd quadmat;

    amrex::Vector<Eigen::MatrixXd> quadmat_bd;

  private:

    //Vandermonde matrix for mapping modes<->quadrature points
    void set_vandermat();

    //Element Matrix and Quadrature Matrix
    void set_ref_element_matrix();

    amrex::Real refMat_phiDphi(int j, const amrex::Vector<amrex::Vector<int>>& idx_map_j,
                              int i, const amrex::Vector<amrex::Vector<int>>& idx_map_i,
                              int dim) const ;
    
    amrex::Real refMat_tphitphi(int j,int i) const;
    
    amrex::Real refMat_tphiDtphi(int j,int i) const;
      
    amrex::Real coefficient_c(int k,int l) const;     

    void set_predictor(const amrex::MultiFab* _U_w,
                       amrex::MultiFab* _H_w);

    void get_H_from_H_w(int M, int N, amrex::MultiFab* _H,
                        amrex::MultiFab* _H_w,
                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    void update_U_w(int lev);

    void update_H_w(int lev);

    template <typename EquationType> 
    void L1Norm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);

    template <typename EquationType> 
    void L2Norm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);   
    
    template <typename EquationType> 
    void LpNorm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                      int _p, amrex::Vector<amrex::Vector<amrex::Real>> quad_pt, int N) const;
    
    solver::Array2D<std::unique_ptr<amrex::FluxRegister>> flux_reg;

    amrex::Vector<amrex::Vector<amrex::Real>> quad_weights_st_bdm;

    amrex::Vector<amrex::Vector<amrex::Real>> quad_weights_st_bdp;

    //Vandermonde matrix
    Eigen::MatrixXd V;
    //  inverse
    Eigen::MatrixXd Vinv;

    //Mass element matrix for ADER-DG corrector
    Eigen::MatrixXd Mk_corr;

    //Stiffness element matrix for ADER-DG corrector
    amrex::Vector<Eigen::MatrixXd> Sk_corr;

    //Mass boundary element matrix for ADER-DG corrector and predictor
    amrex::Vector<Eigen::MatrixXd> Mkbdm;
    amrex::Vector<Eigen::MatrixXd> Mkbdp;

    //Mass element matrix for source term (corrector step)
    Eigen::MatrixXd Mk_corr_src;

    //Mass element matrix for ADER predictor
    Eigen::MatrixXd Mk_h_w;
    Eigen::MatrixXd Mk_h_w_inv;
    Eigen::MatrixXd Mk_pred;

    //Stiffness element matrix for ADER predictor
    amrex::Vector<Eigen::MatrixXd> Sk_pred;
    amrex::Vector<Eigen::MatrixXd> Sk_predVinv;

    //Mass element matrix for source term (predictor step)
    Eigen::MatrixXd Mk_pred_src;
    Eigen::MatrixXd Mk_pred_srcVinv; 

    //RHS temporary for ADER-DG corrector (Np_s components, one per level)
    amrex::Vector<amrex::MultiFab> rhs_corr;

    //RHS temporary for ADER predictor (Np_st components, one per level)
    amrex::Vector<amrex::MultiFab> rhs_pred;

    //ADER predictor vector U(x,t) — (lev,q)
    solver::Array2D<amrex::MultiFab> H;

    //ADER Modal/Nodal predictor vector H_w — (lev,q)
    solver::Array2D<amrex::MultiFab> H_w;

    //ADER predictor vector U(x,t) evaluated at boundary plus (+) b+ — (lev,q)
    solver::Array2D<amrex::MultiFab> H_p;

    //ADER predictor vector U(x,t) evaluated at boundary minus (-) b- — (lev,q)
    solver::Array2D<amrex::MultiFab> H_m;

    //TODO: mybe nested functions ptr dont need to be shared
    //      also mabye can use again CRTP and define them genrally inside Solver

    std::shared_ptr<BasisLegendre> basefunc;  //TODO:maybe doe snot need to be shared

    std::shared_ptr<QuadratureGaussLegendre> quadrule;  //TODO:maybe doe snot need to be shared

    std::shared_ptr<L2ProjInterp> amr_interpolator;

    void DEBUG_print_MFab();
};

//templated methods

template <typename EquationType> 
void AmrDG::set_initial_condition(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, int lev)
{
  
  //Print(*ofs) <<"AmrDG::InitialCondition() "<<lev<<"\n";
  //applies the initial condition to all the solution components modes
  amrex::Vector<amrex::MultiFab *> state_uw(Q);

  for(int q=0; q<Q; ++q){
    state_uw[q] = &(U_w(lev,q));
  }
  
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
    amrex::Vector< amrex::Array4<amrex::Real> > uw(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(*(state_uw[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(*(state_uw[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();  // Include ghost cells for IC

      for(int q=0 ; q<Q; ++q){
        uw[q] = state_uw[q]->array(mfi);
      }

      amrex::ParallelFor(bx,basefunc->Np_s,[&] (int i, int j, int k, int n) noexcept
      {
        for(int q=0; q<Q; ++q){
          uw[q](i,j,k,n) = set_initial_condition_U_w(model_pde,lev,q,n,i, j, k);
        }
      });
    }   
  }
}

template <typename EquationType> 
amrex::Real AmrDG::set_initial_condition_U_w(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,int lev,int q,int n,int i,int j,int k)
{
  
  //project initial condition for solution to initial condition for solution modes         
  amrex::Real sum = 0.0;
  for(int m=0; m<quadrule->qMp_s; ++m) 
  {
    sum+= set_initial_condition_U(model_pde,lev,q,i,j,k,quadrule->xi_ref_quad_s[m])*quadmat(n,m);   
  }
  
  return (sum/(refMat_phiphi(n,basefunc->basis_idx_s,n,basefunc->basis_idx_s)));  
}

template <typename EquationType> 
amrex::Real AmrDG::set_initial_condition_U(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                            int lev,int q,int i,int j,int k, 
                                            const amrex::Vector<amrex::Real>& xi) const
{
  auto _mesh = mesh.lock();

  amrex::Real u_ic;
  u_ic = model_pde->pde_IC(lev,q,i,j,k,xi,_mesh);

  return u_ic;
}

//  ComputeDt, time_integration
template <typename EquationType>
void AmrDG::evolve(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                  const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond)
{

  bool dtn_plt; bool dt_plt; int n; amrex::Real t; std::ostringstream oss;
  int last_progress = 0;  // Track last progress for bar updates
  amrex::Real t_last_regrid = 0.0; // Track last regrid time for dt_regrid
  int n_regrids = 0; // Total number of regrids performed

  auto _mesh = mesh.lock();

  //Set timestep idx and time
  n=0;
  t= 0.0;  

  //Plot initial condition
  dtn_plt =  (dtn_outplt > 0);
  dt_plt = (dt_outplt > 0);
  if(dtn_plt || dt_plt){PlotFile(model_pde,U_w,n, t);}
  
  //Output t=0 norm
  L1Norm_DG_AMR(model_pde);
  L2Norm_DG_AMR(model_pde);
  
  Solver<NumericalMethodType>::set_Dt(model_pde);
  
  if (amrex::ParallelDescriptor::IOProcessor()) {
    if (m_bar) {
        std::cout << "\n";
        m_bar->set_option(indicators::option::PostfixText{"0% Starting simulation..."});
        m_bar->set_progress(0);
        std::cout << std::flush;
    }
  }

  //Set up flux registers for AMR synchronization.
  //The AMR grid hierarchy was already created by InitFromScratch
  //(geometric tagging works on zero data).
  //Ghost cells were synced during IC initialization:
  //  - Projection IC: InterpFromCoarseLevel + FillBoundary in AMR_interpolate_initial_condition
  //  - Analytical IC: AMR_sync_initial_condition (average_fine_coarse + FillBoundary)
  if(_mesh->L > 1){
    AMR_set_flux_registers();
  }

  //Pre-allocate FillPatch temporaries (reused every time step)
  solver::Array2D<amrex::MultiFab> fillpatch_mf;
  fillpatch_mf.resize(_mesh->get_finest_lev()+1, Q);
  for(int l=0; l<=_mesh->get_finest_lev(); ++l){
    for(int q=0; q<Q; ++q){
      fillpatch_mf(l,q).define(U_w(l,q).boxArray(), U_w(l,q).DistributionMap(),
                                basefunc->Np_s, _mesh->nghost);
    }
  }

  ///*
  while(t<T)
  {  
    if (amrex::ParallelDescriptor::IOProcessor()) {
      if (m_bar) {
        oss.str("");
        oss.clear();

        // Calculate progress percentage
        int progress = static_cast<int>((t / T) * 100.0);
        progress = std::clamp(progress, 0, 100);

        // Update the text to show current time vs total time
        oss << std::fixed << std::setprecision(0) << progress << "% "
          << "t = " << std::fixed << std::setprecision(4) << t
          << " / " << T
          << " | Dt = " << std::scientific << std::setprecision(2) << Dt;
        m_bar->set_option(indicators::option::PostfixText{oss.str()});

        // Use tick() to advance the bar visually
        while (last_progress < progress) {
          m_bar->tick();
          last_progress++;
        }
        std::cout << std::flush;
      }
    }

    //Remake existing levels and create new fine levels from coarse
    //Skip in validation mode: tagging is static, so regrid produces the same grid.
    if ((_mesh->L > 1)) //&& !flag_analytical_ic)
    {
      bool do_regrid_dtn = (_mesh->dtn_regrid > 0) && (n % _mesh->dtn_regrid == 0);
      bool do_regrid_dt  = (_mesh->dt_regrid > 0) && (t - t_last_regrid >= _mesh->dt_regrid - 1e-12);

      if(do_regrid_dtn || do_regrid_dt){
        // Snapshot geometry before regrid
        int old_finest = _mesh->get_finest_lev();
        amrex::Vector<amrex::BoxArray> old_ba(old_finest + 1);
        for (int l = 0; l <= old_finest; ++l) {
          old_ba[l] = _mesh->get_BoxArray(l);
        }

        _mesh->regrid(0, t);
        amrex::ParallelDescriptor::Barrier();

        // Check if geometry actually changed
        bool grid_changed = (_mesh->get_finest_lev() != old_finest);
        if (!grid_changed) {
          for (int l = 0; l <= _mesh->get_finest_lev(); ++l) {
            if (_mesh->get_BoxArray(l) != old_ba[l]) {
              grid_changed = true;
              break;
            }
          }
        }

        if (grid_changed) {
          //clear old flux register
          flux_reg.clear();

          //construct new flux register on new grid
          AMR_set_flux_registers();

          //re-allocate FillPatch temporaries on new grid
          fillpatch_mf.resize(_mesh->get_finest_lev()+1, Q);
          for(int l=0; l<=_mesh->get_finest_lev(); ++l){
            for(int q=0; q<Q; ++q){
              fillpatch_mf(l,q).define(U_w(l,q).boxArray(), U_w(l,q).DistributionMap(),
                                        basefunc->Np_s, _mesh->nghost);
            }
          }

          n_regrids++;
        }

        t_last_regrid = t;
      }
    }  
    
    // Advance solution by one time-step.
    Solver<NumericalMethodType>::time_integration(model_pde,bdcond,t);
    
    //limit solution
    //if((t_limit>0) && (n%t_limit==0)){Limiter_w(finest_level);}

    // Gather valid fine cell solutions U_w into valid coarse cells
    Solver<NumericalMethodType>::AMR_average_fine_coarse();   
    
    //Prepare inner ghost cell data for next time step
    //for grids at same level and fine-coarse interface
    //fine grids ghost cells inteprolated from coarse
    for(int l=0; l<=_mesh->get_finest_lev(); ++l){
      for(int q=0 ; q<Q; ++q){
        fillpatch_mf(l,q).setVal(0.0);
      }

      AMR_FillPatch(l, t, &fillpatch_mf(l,0), 0, basefunc->Np_s);

      for(int q=0 ; q<Q; ++q){
        std::swap(U_w(l,q),fillpatch_mf(l,q));
      }
    }
      
    // Update timestep idx and physical time
    n+=1;
    t+=Dt;
    
    //Plotting at pre-specified times
    dtn_plt = (dtn_outplt > 0) && (n % dtn_outplt == 0);
    dt_plt  = (dt_outplt > 0) && (std::abs(std::fmod(t, dt_outplt)) < 1e-02);
    //use as tolerance dt_outplt, i.e same order of magnitude
    if(dtn_plt){PlotFile(model_pde,U_w,n, t);}
    else if(dt_plt){PlotFile(model_pde,U_w,n, t);}

    //Set time-step size
    Solver<NumericalMethodType>::set_Dt(model_pde);
    if(T-t<Dt){Dt = T-t;}    
  }

  if (amrex::ParallelDescriptor::IOProcessor()) {
    if (m_bar && !m_bar->is_completed()) {
        m_bar->set_option(indicators::option::PostfixText{"100% Done"});
        m_bar->set_progress(100);
        m_bar->mark_as_completed();
        std::cout << std::flush;
    }
    m_bar.reset();
    // Restore cursor visibility
    std::cout << "\033[?25h" << std::flush;
    std::cout << "\n";
    Print() << "Total number of time steps: " << n << "\n";
    Print() << "Total number of regrids: " << n_regrids << "\n";
  }
  //*/
  amrex::ParallelDescriptor::Barrier();

  //Output t=T norm
  L1Norm_DG_AMR(model_pde);
  L2Norm_DG_AMR(model_pde);
}

template <typename EquationType>
void AmrDG::time_integration(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                            const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                            amrex::Real time)
{
  auto _mesh = mesh.lock();

  ADER(model_pde,bdcond,time);

  // Add MPI synchronization after time integration
  amrex::ParallelDescriptor::Barrier();
}

template <typename EquationType>
void AmrDG::ADER(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                 const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                  amrex::Real time)
{ 
  //NB:this function  expectes the incoming MFabs (solution) internal ghost cells
  //to be already synchronized. This is ensured by startign from IC that is fully sync
  //and then after everytime-step, sync the updated data
  auto _mesh = mesh.lock();

  if ((_mesh->L > 1))
  {
    // Reset flux registers at beginning of timestep
    for (int l = 1; l <= _mesh->get_finest_lev(); ++l) {
      for(int q=0; q<Q; ++q){
        if (flux_reg(l,q)) {
            flux_reg(l,q)->setVal(0.0);
        }
      }
    }
  }

  //iterate from finest level to coarsest
  for (int l = _mesh->get_finest_lev(); l >= 0; --l){
    //apply BC
    Solver<NumericalMethodType>::FillBoundaryCells(bdcond, &U_w(l,0), l, time);

    //set predictor initial guess
    set_predictor(&U_w(l,0), &H_w(l,0));

    //iteratively find predictor
    int iter=0;
    while(iter<p)
    {
      get_H_from_H_w(quadrule->qMp_st,basefunc->Np_st, &H(l,0), &H_w(l,0),quadrule->xi_ref_quad_st);
      if(model_pde->flag_source_term){
        Solver<NumericalMethodType>::source(l,quadrule->qMp_st,model_pde, &H(l,0), &S(l,0),quadrule->xi_ref_quad_st);
      }

      for(int d = 0; d<AMREX_SPACEDIM; ++d){
        Solver<NumericalMethodType>::flux(l,d,quadrule->qMp_st,model_pde, &H(l,0), &F(l,d,0),quadrule->xi_ref_quad_st);
      }

      //update predictor
      update_H_w(l);

      iter+=1;
    }

    //use found predictor to compute corrector
    get_H_from_H_w(quadrule->qMp_st,basefunc->Np_st, &H(l,0), &H_w(l,0),quadrule->xi_ref_quad_st);
    if(model_pde->flag_source_term){
      Solver<NumericalMethodType>::source(l,quadrule->qMp_st,model_pde, &H(l,0), &S(l,0),quadrule->xi_ref_quad_st);
    }
    for(int d = 0; d<AMREX_SPACEDIM; ++d){

      Solver<NumericalMethodType>::flux(l,d,quadrule->qMp_st,model_pde, &H(l,0), &F(l,d,0),quadrule->xi_ref_quad_st);

      get_H_from_H_w(quadrule->qMp_st_bd,basefunc->Np_st, &H_m(l,0), &H_w(l,0),quadrule->xi_ref_quad_st_bdm[d]);
      Solver<NumericalMethodType>::flux_bd(l,d,quadrule->qMp_st_bd,model_pde, &H_m(l,0), &Fm(l,0), &DFm(l,0),quadrule->xi_ref_quad_st_bdm[d]);

      get_H_from_H_w(quadrule->qMp_st_bd,basefunc->Np_st, &H_p(l,0), &H_w(l,0),quadrule->xi_ref_quad_st_bdp[d]);
      Solver<NumericalMethodType>::flux_bd(l,d,quadrule->qMp_st_bd,model_pde, &H_p(l,0), &Fp(l,0), &DFp(l,0),quadrule->xi_ref_quad_st_bdp[d]);

      Solver<NumericalMethodType>::numflux(l,d,quadrule->qMp_st_bd,basefunc->Np_s, &H_m(l,0), &H_p(l,0), &Fm(l,0), &Fp(l,0), &DFm(l,0), &DFp(l,0));

      if ((_mesh->L > 1))
      {
        for (int q = 0; q < Q; ++q) {
          if (l < _mesh->get_finest_lev() && flux_reg(l+1,0)) {
            flux_reg(l+1,q)->CrseAdd(Fnum_int_c(l,d,q), d,
                        0, 0, static_cast<int>(basefunc->Np_s),
                        -1.0, _mesh->get_Geom(l));
          }

          if (l > 0 && flux_reg(l,0)) {
            flux_reg(l,q)->FineAdd(Fnum_int_f(l,d,q), d,
                          0, 0, static_cast<int>(basefunc->Np_s),
                          1.0);
          }
        }
      }
    }
    //update corrector
    update_U_w(l);
  }

  if ((_mesh->L > 1))
  {
    AMR_flux_correction();
  }
}

template <typename EquationType>
void AmrDG::flux(int lev, int d, int M,
                 const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                 amrex::MultiFab* _U,
                 amrex::MultiFab* _F,
                 const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    // Computes all Q components of the nonlinear flux at the given M interpolation/quadrature points xi
    auto _mesh = mesh.lock();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        amrex::Vector<amrex::Array4<amrex::Real>> flux(Q);
        amrex::Vector<amrex::Array4<const amrex::Real>> u(Q);

#ifdef AMREX_USE_OMP
        for (MFIter mfi(_F[0], MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
#else
        for (MFIter mfi(_F[0], true); mfi.isValid(); ++mfi)
#endif
        {
            const amrex::Box& bx = mfi.growntilebox();

            for (int q = 0; q < Q; ++q) {
                u[q] = _U[q].const_array(mfi);
                flux[q] = _F[q].array(mfi);
            }

            for (int q = 0; q < Q; ++q) {
                amrex::ParallelFor(bx, M, [&](int i, int j, int k, int m) noexcept {
                    flux[q](i, j, k, m) = model_pde->pde_flux(lev, d, q, m, i, j, k, &u, xi[m], _mesh);
                });
            }
        }
    }
}

template <typename EquationType>
void AmrDG::flux_bd(int lev,int d, int M,
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::MultiFab* _U,
                    amrex::MultiFab* _F,
                    amrex::MultiFab* _DF,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  //General function that computes all Q components of the non-linear flux at
  //the given set of M interpolation/quadrature points xi
  auto _mesh = mesh.lock();

  #ifdef AMREX_USE_OMP
  #pragma omp parallel
  #endif
  {
    amrex::Vector< amrex::Array4<amrex::Real> > flux(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > dflux(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > u(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(_F[0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(_F[0],true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        u[q] = _U[q].const_array(mfi);
        flux[q] = _F[q].array(mfi);
        dflux[q] = _DF[q].array(mfi);
      }

      amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
      {
        for(int q=0 ; q<Q; ++q){
          (flux[q])(i,j,k,m) = model_pde->pde_flux(lev,d,q,m,i, j, k, &u, xi[m],_mesh);
          (dflux[q])(i,j,k,m) = model_pde->pde_dflux(lev,d,q,m,i, j, k, &u, xi[m],_mesh);
        }
      });
    }
  }
}

//compute minimum time step size s.t CFL condition is met
template <typename EquationType>
void AmrDG::set_Dt(const std::shared_ptr<ModelEquation<EquationType>>& model_pde)
{
  
  auto _mesh = mesh.lock();

  amrex::Vector<amrex::Real> dt_tmp(_mesh->get_finest_lev()+1);//TODO:proper access to finest_level (in Mesh)

  for (int l = 0; l <= _mesh->get_finest_lev(); ++l)
  {
    const auto dx = _mesh->get_dx(l);

    //compute average mesh size
    amrex::Real dx_avg = 0.0;
    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      dx_avg+=(amrex::Real)dx[d];
    }
    dx_avg /= (amrex::Real)AMREX_SPACEDIM;

    //get solution evaluated at cells
    get_U_from_U_w(quadrule->qMp_s, basefunc->Np_s, &U(l,0), &U_w(l,0), quadrule->xi_ref_quad_s);

    //vector to accumulate all the min dt of all cells of given layer computed by this rank
    amrex::Vector<amrex::Real> rank_min_dt;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  {
      amrex::Vector<amrex::Array4<const amrex::Real>> uc(Q);

    #ifdef AMREX_USE_OMP
      for (MFIter mfi(U(l,0),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
      for (MFIter mfi(U(l,0),true); mfi.isValid(); ++mfi)
    #endif
      {
        const amrex::Box& bx = mfi.tilebox();

        for(int q=0; q<Q; ++q){
          uc[q] = U(l,q).const_array(mfi);
        }

        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          //compute max signal velocity lambda_max(for scalar case is derivative 
          //of flux, for system case is eigenvalue of flux jacobian
          amrex::Real lambda_max = 0.0;
          amrex::Vector<amrex::Real> lambda_d(AMREX_SPACEDIM);
          for(int d = 0; d < AMREX_SPACEDIM; ++d){  
            //compute use cell avg so m==0
            lambda_d[d]= model_pde->pde_cfl_lambda(d,0,i,j,k,&uc);
          }
          //find max signal speed across the dimensions
          auto lambda_max_  = std::max_element(lambda_d.begin(),lambda_d.end());
          lambda_max = static_cast<amrex::Real>(*lambda_max_);         

          //general CFL formulation
          CFL = (1.0/(2.0*(amrex::Real)p+1.0))*(1.0/(amrex::Real)AMREX_SPACEDIM);
          amrex::Real dt_cfl = CFL*(dx_avg/lambda_max);
        
        #ifdef AMREX_USE_OMP
          #pragma omp critical
        #endif
          {
            rank_min_dt.push_back(dt_cfl);
          }
        });         
      }
    }   

    //Find min dt across all cells of layer l
    amrex::Real rank_lev_dt_min= 1.0;
    if (!rank_min_dt.empty()) {
      //compute the min in this rank for this level   
      auto rank_lev_dt_min_ = std::min_element(rank_min_dt.begin(), rank_min_dt.end());

      if (rank_lev_dt_min_ != rank_min_dt.end()) {
        rank_lev_dt_min = static_cast<amrex::Real>(*rank_lev_dt_min_);
      }
    }
  
    //Find min for across MPI processes
    ParallelDescriptor::Barrier();
    ParallelDescriptor::ReduceRealMin(rank_lev_dt_min);//, dt_tmp.size());
    dt_tmp[l] = rank_lev_dt_min;
  }

  //Find min dt across all layers
  amrex::Real dt_min = 1.0;
  if (!dt_tmp.empty()) {
    //min across levels
    auto dt_min_  = std::min_element(dt_tmp.begin(),dt_tmp.end());
    
    if (dt_min_ != dt_tmp.end()) {
      dt_min = (amrex::Real)(*dt_min_);//static_cast<amrex::Real>(*dt_min_); 
    }
  }

  ParallelDescriptor::Barrier();
  ParallelDescriptor::ReduceRealMin(dt_min);
  Dt = c_dt*dt_min;
}

template <typename EquationType>
void AmrDG::source(int lev,int M,
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::MultiFab* _U,
                    amrex::MultiFab* _S,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  auto _mesh = mesh.lock();

  #ifdef AMREX_USE_OMP
  #pragma omp parallel
  #endif
  {
    amrex::Vector< amrex::Array4<amrex::Real> > source(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > u(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(_S[0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(_S[0],true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        u[q] = _U[q].const_array(mfi);
        source[q] = _S[q].array(mfi);
      }

      amrex::ParallelFor(bx,M,[&] (int i, int j, int k, int m) noexcept
      {
        for(int q=0 ; q<Q; ++q){
          (source[q])(i,j,k,m) = model_pde->pde_source(lev,q,m,i,j,k,&u,xi[m],_mesh);
        }
      });
    }
  }
}

template <typename EquationType> 
void AmrDG::LpNorm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                  int _p, amrex::Vector<amrex::Vector<amrex::Real>> quad_pt, int N) const
{
  auto _mesh = mesh.lock();
 
  amrex::Vector<amrex::Vector<amrex::Real>> Lpnorm_multilevel(Q);
  amrex::Vector<amrex::Real> V_level;
 
  for(int l=0; l<=_mesh->get_finest_lev(); ++l)
  {
    amrex::Vector<const amrex::MultiFab *> state_u_h(Q);
    amrex::Vector< amrex::Array4<const amrex::Real>> uh(Q);  
      
    amrex::Vector<amrex::MultiFab> U_h_DG(Q);
    
    for(int q=0; q<Q;++q){
      amrex::BoxArray c_ba = U_w(l,q).boxArray();
      U_h_DG[q].define(c_ba, U_w(l,q).DistributionMap(), basefunc->Np_s, _mesh->nghost);
      amrex::MultiFab::Copy(U_h_DG[q], U_w(l,q), 0, 0, basefunc->Np_s, _mesh->nghost);
    }

    // Get number of cells of full level and intersection level
    amrex::BoxArray c_ba = U_w(l,0).boxArray();
    int N_full = (int)(c_ba.numPts());

    int N_overlap = 0;
    if(l != _mesh->get_finest_lev()){
      amrex::BoxArray f_ba = U_w(l+1,0).boxArray();
      amrex::BoxArray f_ba_c = f_ba.coarsen(_mesh->get_refRatio(l));
      N_overlap = (int)(f_ba_c.numPts());
    }
      
    auto dx = _mesh->get_Geom(l).CellSizeArray();  
    amrex::Real vol = 1.0;
    for(int d = 0; d < AMREX_SPACEDIM; ++d) {
      vol *= dx[d];
    }

    V_level.push_back((amrex::Real)(vol*(amrex::Real)(N_full-N_overlap)));

    // Compute Lp norm on full level
    for(int q=0; q<Q;++q){
      state_u_h[q] = &(U_h_DG[q]);
    }
      
    // Vector to accumulate all the full level norm (reduction sum of all cells norms)
    amrex::Vector<amrex::Real> Lpnorm_full(Q, 0.0);
    amrex::Vector<amrex::Vector<amrex::Real>> Lpnorm_full_tmp(Q);
      
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
      for (MFIter mfi(*(state_u_h)[0],true); mfi.isValid(); ++mfi){
        const amrex::Box& bx_tmp = mfi.tilebox();

        for(int q=0 ; q<Q; ++q){
          uh[q] = state_u_h[q]->const_array(mfi);
        }
            
        if(l != _mesh->get_finest_lev()){
          amrex::BoxArray f_ba = U_w(l+1,0).boxArray();
          amrex::BoxArray ba_c = f_ba.coarsen(_mesh->get_refRatio(l));
          const amrex::BoxList f_ba_lst(ba_c);
            
          // Get complement: boxes NOT covered by finer level
          amrex::BoxList f_ba_lst_compl = complementIn(bx_tmp, f_ba_lst);
                
          amrex::ParallelFor(bx_tmp,[&] (int i, int j, int k) noexcept
          {
            bool is_not_covered = false;
            amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            
            // Check if cell is in complement (NOT covered by finer level)
            for (const amrex::Box& bx : f_ba_lst_compl) {
              if(bx.contains(iv)) {
                is_not_covered = true;
                break;
              }          
            }
              
            // Compute norm only for cells NOT covered by finer level
            if(is_not_covered) {
              for(int q=0 ; q<Q; ++q){
                amrex::Real cell_Lpnorm = 0.0;
                amrex::Real w;
                amrex::Real f;
                  
                for (int m = 0; m < std::pow(N,AMREX_SPACEDIM); ++m){
                  // Quad weights for each quadrature point
                  w = 1.0;
                  for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
                    w *= 2.0/std::pow(std::assoc_legendre(N,1,quad_pt[m][d_]),2);
                  }

                  amrex::Real u_h = 0.0;          
                  for (int n = 0; n < basefunc->Np_s; ++n){  
                    u_h += uh[q](i,j,k,n)*(basefunc->phi_s(n,basefunc->basis_idx_s,quad_pt[m]));
                  }
                        
                  amrex::Real u = 0.0;
                  u = set_initial_condition_U(model_pde,l,q,i,j,k, quad_pt[m]);
                    
                  f = std::pow(std::abs(u-u_h),(amrex::Real)_p);
                  cell_Lpnorm += (f*w);
                }
                amrex::Real coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
#pragma omp critical
                {
                  Lpnorm_full_tmp[q].push_back(cell_Lpnorm*coeff);  
                }
              }          
            }
          });
        }  
        else {
          // Finest level - include all cells
          amrex::ParallelFor(bx_tmp,[&] (int i, int j, int k) noexcept
          {
            for(int q=0 ; q<Q; ++q){
              amrex::Real cell_Lpnorm = 0.0;
              amrex::Real w;
              amrex::Real f;
                
              for (int m = 0; m < std::pow(N,AMREX_SPACEDIM); ++m){
                // Quad weights for each quadrature point
                w = 1.0;
                for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
                  w *= 2.0/std::pow(std::assoc_legendre(N,1,quad_pt[m][d_]),2);
                }
                  
                amrex::Real u_h = 0.0;          
                for (int n = 0; n < basefunc->Np_s; ++n){  
                  u_h += uh[q](i,j,k,n)*(basefunc->phi_s(n,basefunc->basis_idx_s,quad_pt[m]));
                }
                      
                amrex::Real u = 0.0;
                u = set_initial_condition_U(model_pde,l,q,i,j,k, quad_pt[m]);
                      
                f = std::pow(std::abs(u-u_h),(amrex::Real)_p);
                cell_Lpnorm += (f*w);
              }
              amrex::Real coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
#pragma omp critical
              {
                Lpnorm_full_tmp[q].push_back(cell_Lpnorm*coeff);  
              }
            }      
          });    
        }  
      }
    }
    
    for(int q=0 ; q<Q; ++q){
      amrex::Real global_Lpnorm = 0.0;
      global_Lpnorm = std::accumulate(Lpnorm_full_tmp[q].begin(),
                                      Lpnorm_full_tmp[q].end(), 0.0);
                                      
      ParallelDescriptor::ReduceRealSum(global_Lpnorm);
      Lpnorm_full[q] = global_Lpnorm;
    } 
      
    for(int q=0 ; q<Q; ++q){
      Lpnorm_multilevel[q].push_back((amrex::Real)Lpnorm_full[q]);      
    }      
  }
    
  amrex::Real V_amr = (amrex::Real)std::accumulate(V_level.begin(),V_level.end(), 0.0);  
  for(int q=0 ; q<Q; ++q){
    amrex::Real Lpnorm = std::accumulate(Lpnorm_multilevel[q].begin(),
                                         Lpnorm_multilevel[q].end(), 0.0);
                                             
    Lpnorm = std::pow(Lpnorm/V_amr, 1.0/(amrex::Real)_p);
    Print().SetPrecision(17) << "--multilevel--" << "\n";
    Print().SetPrecision(17) << "L" << _p << " error norm:  " << Lpnorm << " | "
                           << "DG Order:  " << p+1 << " | solution component: " << q << "\n";
  }  
}

template <typename EquationType> 
void AmrDG::L1Norm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde)
{
  LpNorm_DG_AMR(model_pde,1, quadrule->xi_ref_quad_s,quadrule->qMp_1d);
}

template <typename EquationType> 
void AmrDG::L2Norm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde) 
{ 
  //TODO:actually could generalize it to p points
  //Generate 2*(p+1) quadrature points in 1D
  
  int N = 2*(quadrule->qMp_1d);
  amrex::Vector<amrex::Real> GLquadpts;
  amrex::Real xiq = 0.0;
  amrex::Real theta = 0.0;
  for(int i=1; i<= (int)(N/2); ++i)
  {
    theta = M_PI*(i - 0.25)/((double)N + 0.5);
    if((1<=i) && (i<= (int)((1.0/3.0)*(double)N))){
      xiq = (1-0.125*(1.0/std::pow(N,2))+0.125*(1.0/std::pow(N,3))
            -(1.0/384.0)*(1.0/std::pow(N,4))*(39.0-28.0*(1.0/std::pow(std::sin(theta),2))))
            *std::cos(theta);
    }
    else if((i>(int)((1.0/3.0)*(double)N)) && (i<= (int)((double)N/2))){
      xiq = (1.0-(1.0/(8.0*std::pow((double)N,2)))
          +(1.0/(8.0*std::pow((double)N,3))))*std::cos(theta);
    }
    quadrule->NewtonRhapson(xiq, N);
    GLquadpts.push_back(xiq);   
    GLquadpts.push_back(-xiq);  
  }

  //TODO: below will always be zero right?, therefore could just do GLquadpts.push_back(0.0);   
  if(N%2!=0)//if odd number, then i=1,...,N/2 will miss one value
  {
    int i = (N/2)+1;
    theta = M_PI*(i - 0.25)/((double)N + 0.5);
    xiq = (1.0-(1.0/(8.0*std::pow((double)N,2)))
          +(1.0/(8.0*std::pow((double)N,3))))*std::cos(theta);
    quadrule->NewtonRhapson(xiq, N);
    GLquadpts.push_back(xiq);   
  }//TODO: dont rememebr why is it different than in AmrDG_Quadrature
  
  amrex::Vector<amrex::Vector<amrex::Real>> GLquadptsL2norm; 
  GLquadptsL2norm.resize((int)std::pow(N,AMREX_SPACEDIM),
                        amrex::Vector<amrex::Real> (AMREX_SPACEDIM));
                        
  #if (AMREX_SPACEDIM == 1)
  for(int i=0; i<N;++i)
  {
    GLquadptsL2norm[i][0]=GLquadpts[i];
  }
  #elif (AMREX_SPACEDIM == 2)
  for(int i=0; i<N;++i){
    for(int j=0; j<N;++j){
        GLquadptsL2norm[j+N*i][0]=GLquadpts[i];
        GLquadptsL2norm[j+N*i][1]=GLquadpts[j]; 
    }
  }
  #elif (AMREX_SPACEDIM == 3)
  for(int i=0; i<N;++i){
    for(int j=0; j<N;++j){
      for(int k=0; k<N;++k){
        GLquadptsL2norm[k+N*j+N*N*i][0]=GLquadpts[i];
        GLquadptsL2norm[k+N*j+N*N*i][1]=GLquadpts[j]; 
        GLquadptsL2norm[k+N*j+N*N*i][2]=GLquadpts[k]; 
      }
    }
  }
  #endif 
  
  LpNorm_DG_AMR(model_pde,2, GLquadptsL2norm,2*(quadrule->qMp_1d));
}

#endif