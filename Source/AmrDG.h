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
    //type alias to improve readibility
    using NumericalMethodType = AmrDG;

    AmrDG()  = default;

    ~AmrDG();

    void settings(int _p, amrex::Real _T) {
      p = _p;
      T = _T;
    }

    void init();

    void init_bc(amrex::Vector<amrex::Vector<amrex::BCRec>>& bc, int& n_comp);

    template <typename EquationType>
    void evolve(std::shared_ptr<ModelEquation<EquationType>> model_pde,
                std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>> bdcond);

    template <typename EquationType>
    void time_integration(std::shared_ptr<ModelEquation<EquationType>> model_pde);

    template <typename EquationType>
    void ADER(std::shared_ptr<ModelEquation<EquationType>> model_pde);

    template <typename EquationType>
    void set_Dt(std::shared_ptr<ModelEquation<EquationType>> model_pde);

    void set_init_data_system(int lev,const BoxArray& ba,
                              const DistributionMapping& dm);

    void set_init_data_component(int lev,const BoxArray& ba,
                                const DistributionMapping& dm, int q);

    void get_U_from_U_w(int M, int N,amrex::Vector<amrex::MultiFab>* U_ptr,
                        amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi);
    
    template <typename EquationType>                    
    void source(int lev,int M, 
                std::shared_ptr<ModelEquation<EquationType>> model_pde,
                amrex::Vector<amrex::MultiFab>* U_ptr,
                amrex::Vector<amrex::MultiFab>* S_ptr,
                const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    template <typename EquationType>
    void flux(int lev,int d, int M, 
              std::shared_ptr<ModelEquation<EquationType>> model_pde,
              amrex::Vector<amrex::MultiFab>* U_ptr,
              amrex::Vector<amrex::MultiFab>* F_ptr,
              const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    template <typename EquationType>
    void flux_bd(int lev,int d, int M, 
                std::shared_ptr<ModelEquation<EquationType>> model_pde,
                amrex::Vector<amrex::MultiFab>* U_ptr,
                amrex::Vector<amrex::MultiFab>* F_ptr,
                amrex::Vector<amrex::MultiFab>* DF_ptr,
                const amrex::Vector<amrex::Vector<amrex::Real>>& xi);
  
    void numflux_integral(int lev,int d,int M, int N,
                          amrex::Vector<amrex::MultiFab>* U_ptr_m, 
                          amrex::Vector<amrex::MultiFab>* U_ptr_p,
                          amrex::Vector<amrex::MultiFab>* F_ptr_m,
                          amrex::Vector<amrex::MultiFab>* F_ptr_p,
                          amrex::Vector<amrex::MultiFab>* DF_ptr_m,
                          amrex::Vector<amrex::MultiFab>* DF_ptr_p);

    amrex::Real numflux(int d, int m,int i, int j, int k, 
                amrex::Array4<const amrex::Real> up, 
                amrex::Array4<const amrex::Real> um, 
                amrex::Array4<const amrex::Real> fp,
                amrex::Array4<const amrex::Real> fm,  
                amrex::Array4<const amrex::Real> dfp,
                amrex::Array4<const amrex::Real> dfm);
    
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

    //Element Matrix and Quadrature Matrix
    void set_ref_element_matrix();

    amrex::Real refMat_phiphi(int i,int j, bool is_predictor, bool is_mixed_nmodes) const;

    amrex::Real refMat_phiDphi(int i,int j, int dim) const;   
    
    amrex::Real refMat_tphitphi(int i,int j) const;
    
    amrex::Real refMat_tphiDtphi(int i,int j) const;
      
    amrex::Real coefficient_c(int k,int l) const;     

    int kroneckerDelta(int a, int b) const;

    void set_predictor(const amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                      amrex::Vector<amrex::MultiFab>* H_w_ptr);
    
    void get_H_from_H_w(int M, int N,amrex::Vector<amrex::MultiFab>* H_ptr,
                        amrex::Vector<amrex::MultiFab>* H_w_ptr, 
                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    void update_U_w(int lev);

    void update_H_w(int lev);
    
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

//templated methods

//  ComputeDt, time_integration
template <typename EquationType>
void AmrDG::evolve(std::shared_ptr<ModelEquation<EquationType>> model_pde,
                  std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>> bdcond)
{

  
  int n=0;
  amrex::Real t= 0.0;  
  //if(t_outplt>0){PlotFile(0, t);}
  //NormDG();
  
  set_Dt(model_pde);
  //Print(*ofs).SetPrecision(6)<<"time: "<< t<<" | time step: "<<n<<" | step size: "<< dt<<"\n";
  //Print(*ofs)<<"------------------------------------------------"<<"\n";
  
  //time stepping
  while(t<T)
  {  
    /*
    if ((max_level > 0) && (n>0))
    {
      if((t_regrid > 0) && (n % t_regrid == 0)){
        regrid(0, t);
      }
    }  
    */
    //Print(*ofs) << "ADER Time Integraton"<< "\n";
    //advance solution by one time-step.
    time_integration(model_pde);

    //limit solution
    //if((t_limit>0) && (n%t_limit==0)){Limiter_w(finest_level);}
    //gather valid fine cell solutions U_w into valid coarse cells
    //AverageFineToCoarse();   
    
    /*
    //prepare data for next time step
    for(int l=0; l<=finest_level; ++l){
      for(int q=0; q<Q; ++q){ 
        //FillPatch(l, t, U_w[l][q], 0, Np,q); 
        U_w[l][q].FillBoundary(geom[l].periodicity());
        if(l>0){FillPatchGhostFC(l,0,q);}
        //FillBoundary will be repeated also for FillPatchGHost, but there there
        //is not info about periodic BC
      }
    }
    */
    n+=1;
    t+=Dt;
    //Print(*ofs).SetPrecision(6)<<"time: "<< t<<" | time step: "<<n<<" | step size: "<< dt<<"\n";
    //Print(*ofs)<<"------------------------------------------------"<<"\n";
    
    //if((t_outplt>0) && (n%t_outplt==0)){PlotFile(n, t);} 
    
    set_Dt(model_pde);
    if(T-t<Dt){Dt = T-t;}    
  }
  
  //NormDG();

}

template <typename EquationType>
void AmrDG::time_integration(std::shared_ptr<ModelEquation<EquationType>> model_pde)
{
  ADER(model_pde);
}

template <typename EquationType>
void AmrDG::ADER(std::shared_ptr<ModelEquation<EquationType>> model_pde)
{
  for (int l = mesh->get_finest_lev(); l >= 0; --l){
    //apply BC
    //FillBoundaryCells(&(U_w[l]), l);
    
    //set predictor initial guess
    set_predictor(&(U_w[l]), &(H_w[l]));  

    //iteratively find predictor
    int iter=0;    
    while(iter<p)
    { 
      if(model_pde->flag_source_term){
        get_H_from_H_w(quadrule->qMp_st,basefunc->Np_st,&(H[l]),&(H_w[l]),quadrule->xi_ref_quad_st);
        source(l,quadrule->qMp_st,model_pde,&(H[l]),&(S[l]),quadrule->xi_ref_quad_st);
      }  
      for(int d = 0; d<AMREX_SPACEDIM; ++d){
        //if source ===True, this is duplicate, could avoid redoit
        get_H_from_H_w(quadrule->qMp_st,basefunc->Np_st,&(H[l]),&(H_w[l]),quadrule->xi_ref_quad_st);
        flux(l,d,quadrule->qMp_st,model_pde,&(H[l]),&(F[l][d]),quadrule->xi_ref_quad_st);
      }   

      //update predictor
      update_H_w(l);

      iter+=1;
    }
    
    //use found predictor to compute corrector
    if(model_pde->flag_source_term){
      get_H_from_H_w(quadrule->qMp_st,basefunc->Np_st,&(H[l]),&(H_w[l]),quadrule->xi_ref_quad_st);
      source(l,quadrule->qMp_st,model_pde,&(H[l]),&(S[l]),quadrule->xi_ref_quad_st);
    }    
    
    for(int d = 0; d<AMREX_SPACEDIM; ++d){
      get_H_from_H_w(quadrule->qMp_st,basefunc->Np_st,&(H[l]),&(H_w[l]),quadrule->xi_ref_quad_st);
      flux(l,d,quadrule->qMp_st,model_pde,&(H[l]),&(F[l][d]),quadrule->xi_ref_quad_st);

      get_H_from_H_w(quadrule->qMp_st_bd,basefunc->Np_st,&(H_m[l][d]),&(H_w[l]),quadrule->xi_ref_quad_st_bdm[d]);
      flux_bd(l,d,quadrule->qMp_st_bd,model_pde,&(H_m[l][d]),&(Fm[l][d]),&(DFm[l][d]),quadrule->xi_ref_quad_st_bdm[d]);

      get_H_from_H_w(quadrule->qMp_st_bd,basefunc->Np_st,&(H_p[l][d]),&(H_w[l]),quadrule->xi_ref_quad_st_bdp[d]);
      flux_bd(l,d,quadrule->qMp_st_bd,model_pde,&(H_p[l][d]),&(Fp[l][d]),&(DFp[l][d]),quadrule->xi_ref_quad_st_bdp[d]);

      numflux_integral(l,d,quadrule->qMp_st_bd,basefunc->Np_s,&(H_m[l][d]),&(H_p[l][d]),&(Fm[l][d]),&(Fp[l][d]),&(DFm[l][d]),&(DFp[l][d]));
    } 
      
    //average fine to coarse interface integral numerical flux for conservation
    //AverageFineToCoarseFlux(l);
    
    //update corrector
    update_U_w(l);    
  }
  
}

//compute minimum time step size s.t CFL condition is met
template <typename EquationType>
void AmrDG::set_Dt(std::shared_ptr<ModelEquation<EquationType>> model_pde)
{
  
  amrex::Vector<amrex::Real> dt_tmp(mesh->get_finest_lev()+1);//TODO:proper access to finest_level (in Mesh)

  for (int l = 0; l <= mesh->get_finest_lev(); ++l)
  {
    const auto dx = mesh->get_dx(l);
    //compute average mesh size
    amrex::Real dx_avg = 0.0;
    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      dx_avg+=((amrex::Real)dx[d]/(amrex::Real)AMREX_SPACEDIM);
    }
      
    //evaluate modes at cell center pt
    //1 quadrature poitn to use for evaluation
    get_U_from_U_w(1, basefunc->Np_s,&(U_center[l]),&(U_w[l]), quadrule->xi_ref_quad_s_center);
           
    //vector to accumulate all the min dt computed by this rank
    amrex::Vector<amrex::Real> rank_min_dt;

    amrex::Vector<const amrex::MultiFab *> state_uc(Q);

    for(int q=0; q<Q; ++q){
      state_uc[q]= &(U_center[l][q]);
    } 
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
      amrex::Vector<const amrex::FArrayBox *> fab_uc(Q);
      amrex::Vector<amrex::Array4<const amrex::Real>> uc(Q);
    
      #ifdef AMREX_USE_OMP  
      for (MFIter mfi(U_center[l][0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
      #else
      for (MFIter mfi(U_center[l][0],true); mfi.isValid(); ++mfi)
      #endif 
      {
        const amrex::Box& bx = mfi.tilebox();
        
        for(int q=0; q<Q; ++q){
          fab_uc[q] = state_uc[q]->fabPtr(mfi);
          uc[q] =fab_uc[q]->const_array();
        }              
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          //compute max signal velocity lambda_max(for scalar case is derivative 
          //of flux, for system case is eigenvalue of flux jacobian
          amrex::Real lambda_max = 0.0;

          amrex::Vector<amrex::Real> lambda_d(AMREX_SPACEDIM);
          for(int d = 0; d < AMREX_SPACEDIM; ++d){  
            //compute at cell center so m==0
            lambda_d.push_back(model_pde->pde_cfl_lambda(d,0,i,j,k,&uc));
          }
          //find max signal speed across the dimensions
          auto lambda_max_  = std::max_element(lambda_d.begin(),lambda_d.end());
          lambda_max = static_cast<amrex::Real>(*lambda_max_);         

          //general CFL formulation
          amrex::Real CFL = (1.0/(2.0*(amrex::Real)p+1.0))*(1.0/(amrex::Real)AMREX_SPACEDIM);
          amrex::Real dt_cfl = CFL*(dx_avg/lambda_max);
                        
          #pragma omp critical
          {
            rank_min_dt.push_back(dt_cfl);
          }
        });         
      }
    }   
    amrex::Real rank_lev_dt_min= 1.0;
    if (!rank_min_dt.empty()) {
      //compute the min in this rank for this level   
      auto rank_lev_dt_min_ = std::min_element(rank_min_dt.begin(), rank_min_dt.end());

      if (rank_lev_dt_min_ != rank_min_dt.end()) {
        rank_lev_dt_min = static_cast<amrex::Real>(*rank_lev_dt_min_);
      }
    }
    ParallelDescriptor::ReduceRealMin(rank_lev_dt_min);//, dt_tmp.size());
    dt_tmp[l] = rank_lev_dt_min;
    
  }
  amrex::Real dt_min = 1.0;
  if (!dt_tmp.empty()) {
    //min across levels
    auto dt_min_  = std::min_element(dt_tmp.begin(),dt_tmp.end());
    
    if (dt_min_ != dt_tmp.end()) {
      dt_min = (amrex::Real)(*dt_min_);//static_cast<amrex::Real>(*dt_min_); 
    }
  }
  ParallelDescriptor::ReduceRealMin(dt_min);
  Dt = dt_min; 
  
}

template <typename EquationType>
void AmrDG::source(int lev,int M, 
                    std::shared_ptr<ModelEquation<EquationType>> model_pde,
                    amrex::Vector<amrex::MultiFab>* U_ptr,
                    amrex::Vector<amrex::MultiFab>* S_ptr,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  amrex::Vector<amrex::MultiFab *> state_source(Q);
  amrex::Vector<const amrex::MultiFab *> state_u(Q);

  for(int q=0 ; q<Q; ++q){
  state_u[q] = &((*U_ptr)[q]); 
  state_source[q] = &((*S_ptr)[q]); 
  }

  #ifdef AMREX_USE_OMP
  #pragma omp parallel 
  #endif
  { 
    amrex::Vector<amrex::FArrayBox *> fab_source(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > source(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_u(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > u(Q);

    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_source[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_source[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        fab_u[q] = state_u[q]->fabPtr(mfi);
        fab_source[q]=&((*(state_source[q]))[mfi]);

        u[q] = fab_u[q]->const_array();
        source[q]=(*(fab_source[q])).array();
      } 

      for(int q=0 ; q<Q; ++q){
        amrex::ParallelFor(bx,M,[&] (int i, int j, int k, int m) noexcept
        {
          (source[q])(i,j,k,m) = model_pde->pde_source(lev,q,m,i,j,k,&u,xi[m]);  
        });
      }
    }
  } 
}

template <typename EquationType>
void AmrDG::flux(int lev,int d, int M, 
                std::shared_ptr<ModelEquation<EquationType>> model_pde,
                amrex::Vector<amrex::MultiFab>* U_ptr,
                amrex::Vector<amrex::MultiFab>* F_ptr,
                const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{ 
  //General function that computes all Q components of the non-linear flux at 
  //the given set of M interpolation/quadrature points xi

  amrex::Vector<amrex::MultiFab *> state_flux(Q);
  amrex::Vector<amrex::MultiFab *> state_dflux(Q);
  amrex::Vector<const amrex::MultiFab *> state_u(Q);

  for(int q=0 ; q<Q; ++q){
    state_u[q] = &((*U_ptr)[q]); 
    state_flux[q] = &((*F_ptr)[q]); 
  }

  #ifdef AMREX_USE_OMP
  #pragma omp parallel 
  #endif
  {
    amrex::Vector<amrex::FArrayBox *> fab_flux(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > flux(Q);
    amrex::Vector<amrex::FArrayBox *> fab_dflux(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > dflux(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_u(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > u(Q);

    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_flux[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_flux[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        fab_u[q] = state_u[q]->fabPtr(mfi);
        fab_flux[q]=&((*(state_flux[q]))[mfi]);
        fab_dflux[q]=&((*(state_dflux[q]))[mfi]);

        u[q] = fab_u[q]->const_array();
        flux[q]=(*(fab_flux[q])).array();
      } 
      for(int q=0 ; q<Q; ++q){
        amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
        {              
          (flux[q])(i,j,k,m) =  model_pde->pde_flux(lev,d,q,m,i, j, k, &u, xi[m]);
        });
      }
    }
  }
}

template <typename EquationType>
void AmrDG::flux_bd(int lev,int d, int M, 
                    std::shared_ptr<ModelEquation<EquationType>> model_pde,
                    amrex::Vector<amrex::MultiFab>* U_ptr,
                    amrex::Vector<amrex::MultiFab>* F_ptr,
                    amrex::Vector<amrex::MultiFab>* DF_ptr,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{ 
  //General function that computes all Q components of the non-linear flux at 
  //the given set of M interpolation/quadrature points xi

  amrex::Vector<amrex::MultiFab *> state_flux(Q);
  amrex::Vector<amrex::MultiFab *> state_dflux(Q);
  amrex::Vector<const amrex::MultiFab *> state_u(Q);

  for(int q=0 ; q<Q; ++q){
    state_u[q] = &((*U_ptr)[q]); 
    state_flux[q] = &((*F_ptr)[q]); 
    state_dflux[q] = &((*DF_ptr)[q]);
  }

  #ifdef AMREX_USE_OMP
  #pragma omp parallel 
  #endif
  {
    amrex::Vector<amrex::FArrayBox *> fab_flux(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > flux(Q);
    amrex::Vector<amrex::FArrayBox *> fab_dflux(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > dflux(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_u(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > u(Q);

    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_flux[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_flux[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        fab_u[q] = state_u[q]->fabPtr(mfi);
        fab_flux[q]=&((*(state_flux[q]))[mfi]);
        fab_dflux[q]=&((*(state_dflux[q]))[mfi]);

        u[q] = fab_u[q]->const_array();
        flux[q]=(*(fab_flux[q])).array();
        dflux[q]=(*(fab_dflux[q])).array();
      } 
      for(int q=0 ; q<Q; ++q){
        amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
        {              
        (flux[q])(i,j,k,m) = model_pde->pde_flux(lev,d,q,m,i, j, k, &u, xi[m]);
        (dflux[q])(i,j,k,m) = model_pde->pde_dflux(lev,d,q,m,i, j, k, &u, xi[m]);
        });
      }
    }
  }
}
























///////////////////////////////////////////////////////////////////////////


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