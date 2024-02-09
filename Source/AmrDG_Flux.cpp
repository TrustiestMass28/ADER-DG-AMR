#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>
#include <cmath>
#include <math.h>
#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include "AmrDG.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

void AmrDG::Predictor_set(const amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                          amrex::Vector<amrex::MultiFab>* H_w_ptr)
{
  amrex::Vector<const amrex::MultiFab *> state_u_w(Q);   
  amrex::Vector<amrex::MultiFab *> state_h_w(Q); 

  for(int q=0; q<Q; ++q){
    state_u_w[q]=&((*U_w_ptr)[q]);
    state_h_w[q]=&((*H_w_ptr)[q]); 
  } 

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  { 
    amrex::Vector<amrex::FArrayBox *> fab_h_w(Q);
    amrex::Vector<amrex::Array4<amrex::Real>> hw(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_u_w(Q);
    amrex::Vector<amrex::Array4<const amrex::Real>> uw(Q);   
    
    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_h_w[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(*(state_h_w[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();
      
      for(int q=0 ; q<Q; ++q){      
        fab_u_w[q] = state_u_w[q]->fabPtr(mfi);
        fab_h_w[q]= &((*(state_h_w[q]))[mfi]);
        
        uw[q] = fab_u_w[q]->const_array();
        hw[q]=(*(fab_h_w[q])).array();
      }
      
      for(int q=0 ; q<Q; ++q){
        amrex::ParallelFor(bx, mNp,[&] (int i, int j, int k, int n) noexcept
        {   
          if(n<Np)
          {
            amrex::Real tmp = (uw[q])(i,j,k,n); 
            (hw[q])(i,j,k,n)=tmp;  
          } 
          
          else
          {
            (hw[q])(i,j,k,n)=0.0;
          }                
        }); 
      }      
    }  
  }  
}

void AmrDG::get_U_from_U_w(int c, amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                            amrex::Vector<amrex::MultiFab>* U_ptr,
                            amrex::Vector<amrex::Real> xi, bool is_predictor)
{
  //computes the sum of modes and respective basis function evaluated at specified location

  amrex::Vector<const amrex::MultiFab *> state_u_w(Q);   
  amrex::Vector<amrex::MultiFab *> state_u(Q); 
 
  for(int q=0; q<Q; ++q){
    state_u_w[q]=&((*U_w_ptr)[q]);    
    state_u[q]=&((*U_ptr)[q]);  
  } 
  
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  {
    amrex::Vector<const amrex::FArrayBox *> fab_u_w(Q);
    amrex::Vector< amrex::Array4<const amrex::Real>> uw(Q);  
    amrex::Vector<amrex::FArrayBox *> fab_u(Q);
    amrex::Vector<amrex::Array4<amrex::Real>> u(Q);  
  
    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_u_w[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_u_w[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

        
      for(int q=0 ; q<Q; ++q){
        fab_u_w[q] = state_u_w[q]->fabPtr(mfi);
        uw[q] = fab_u_w[q]->const_array();
             
        fab_u[q]=&((*(state_u[q]))[mfi]);
        u[q]=(*(fab_u[q])).array();
      } 
      for(int q=0 ; q<Q; ++q){
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {(u[q])(i,j,k,c)=0.0;}); 
        if(is_predictor == true){
          amrex::ParallelFor(bx,mNp,[&] (int i, int j, int k,int n) noexcept
          { 
            (u[q])(i,j,k,c)+=((uw[q])(i,j,k,n)*modPhi(n, xi));
          });            
        }
        else if(is_predictor==false){ 
          amrex::ParallelFor(bx,Np,[&] (int i, int j, int k,int n) noexcept
          { 
            (u[q])(i,j,k,c)+=((uw[q])(i,j,k,n)*modPhi(n, xi));
          });  
        }        
      }      
    }  
  }
}

void AmrDG::Source(int lev,int M,
                  amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                  amrex::Vector<amrex::MultiFab>* U_ptr,
                  amrex::Vector<amrex::MultiFab>* S_ptr,
                  amrex::Vector<amrex::Vector<amrex::Real>> xi, 
                  bool is_predictor)
{
  for(int m = 0; m<M ; ++m){
    get_U_from_U_w(m,U_w_ptr, U_ptr, xi[m],is_predictor);
  }

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
          (source[q])(i,j,k,m) = PhysicalSource(lev,q,m,i,j,k,&u,xi[m]);                  
        });
      }
    }
  } 
}

void AmrDG::Flux(int lev,int d, int M, 
                amrex::Vector<amrex::MultiFab>* U_w_ptr, 
                amrex::Vector<amrex::MultiFab>* U_ptr,
                amrex::Vector<amrex::MultiFab>* F_ptr,
                amrex::Vector<amrex::MultiFab>* DF_ptr,
                amrex::Vector<amrex::Vector<amrex::Real>> xi,
                bool flag_bd, bool is_predictor)
{ 
  //General function that computes all Q components of the non-linear flux at 
  //the given set of M interpolation/quadrature points xi
  for(int m = 0; m<M ; ++m){
    get_U_from_U_w(m,U_w_ptr, U_ptr, xi[m], is_predictor);
  }

  amrex::Vector<amrex::MultiFab *> state_flux(Q);
  amrex::Vector<amrex::MultiFab *> state_dflux(Q);
  amrex::Vector<const amrex::MultiFab *> state_u(Q);

  for(int q=0 ; q<Q; ++q){
    state_u[q] = &((*U_ptr)[q]); 
    state_flux[q] = &((*F_ptr)[q]); 
    if(flag_bd){state_dflux[q] = &((*DF_ptr)[q]);}    
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
        if(flag_bd){fab_dflux[q]=&((*(state_dflux[q]))[mfi]);}
        
        u[q] = fab_u[q]->const_array();
        flux[q]=(*(fab_flux[q])).array();
        if(flag_bd){dflux[q]=(*(fab_dflux[q])).array();}
      } 
      for(int q=0 ; q<Q; ++q){
        amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
        {              
          (flux[q])(i,j,k,m) = PhysicalFlux(lev,d,q,m,i, j, k, &u, xi[m]);
          if(flag_bd){(dflux[q])(i,j,k,m) = DPhysicalFlux(lev,d,q,m,i, j, k, &u, xi[m]);}            
        });
      }
    }
  }
}

void AmrDG::InterfaceNumFlux(int lev,int d,int M, 
                            amrex::Vector<amrex::MultiFab>* U_ptr_m, 
                            amrex::Vector<amrex::MultiFab>* U_ptr_p)
{
  //U_ptr_m, U_ptr_p passed as arguments because if use ADERwe pass predictor
  //if use RK methods pass the solution.
  
  auto const dx = geom[lev].CellSizeArray(); 
  amrex::Real dvol = 1.0;
  for(int _d = 0; _d < AMREX_SPACEDIM; ++_d){
    if(_d!=d){dvol*=dx[_d];}
  }

  //computes the numerical flux at the plus interface of a cell, i.e at idx i+1/2 
  amrex::Vector<amrex::MultiFab *> state_fnum(Q); 
  amrex::Vector<amrex::MultiFab *> state_fnumm_int(Q);
  amrex::Vector<amrex::MultiFab *> state_fnump_int(Q);
  
  amrex::Vector<const amrex::MultiFab *> state_fm(Q);
  amrex::Vector<const amrex::MultiFab *> state_fp(Q); 
  amrex::Vector<const amrex::MultiFab *> state_dfm(Q);
  amrex::Vector<const amrex::MultiFab *> state_dfp(Q);
  amrex::Vector<const amrex::MultiFab *> state_um(Q);
  amrex::Vector<const amrex::MultiFab *> state_up(Q);

  for(int q=0 ; q<Q; ++q){
    state_fnum[q] = &(Fnum[lev][d][q]); 
    state_fnumm_int[q] = &(Fnumm_int[lev][d][q]); 
    state_fnump_int[q] = &(Fnump_int[lev][d][q]); 
    state_fm[q] = &(Fm[lev][d][q]); 
    state_fp[q] = &(Fp[lev][d][q]); 
    state_dfm[q] = &(DFm[lev][d][q]); 
    state_dfp[q] = &(DFp[lev][d][q]);     
    state_um[q] = &((*U_ptr_m)[q]); 
    state_up[q] = &((*U_ptr_p)[q]); 
  }
    
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
    amrex::Vector<amrex::FArrayBox *> fab_fnum(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnum(Q);
    amrex::Vector<amrex::FArrayBox *> fab_fnumm_int(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnumm_int(Q);
    amrex::Vector<amrex::FArrayBox *> fab_fnump_int(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > fnump_int(Q);
    
    amrex::Vector<const amrex::FArrayBox *> fab_fm(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > fm(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_fp(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > fp(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_dfm(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > dfm(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_dfp(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > dfp(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_um(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > um(Q);
    amrex::Vector<const amrex::FArrayBox *> fab_up(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > up(Q);
    
    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_fnum[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_fnum[0]),true); mfi.isValid(); ++mfi)
    #endif    
    {
      //externally grown tilebox
      const amrex::Box& bx = mfi.tilebox();
      
      for(int q=0 ; q<Q; ++q){
        fab_fnum[q]=&((*(state_fnum[q]))[mfi]);
        fab_fnumm_int[q]=&((*(state_fnumm_int[q]))[mfi]);
        fab_fnump_int[q]=&((*(state_fnump_int[q]))[mfi]);        
        fab_fm[q] = state_fm[q]->fabPtr(mfi);
        fab_fp[q] = state_fp[q]->fabPtr(mfi);
        fab_dfm[q] = state_dfm[q]->fabPtr(mfi);
        fab_dfp[q] = state_dfp[q]->fabPtr(mfi);
        fab_um[q] = state_um[q]->fabPtr(mfi);
        fab_up[q] = state_up[q]->fabPtr(mfi);
        
        fnum[q]=(*(fab_fnum[q])).array();
        fnumm_int[q]=(*(fab_fnumm_int[q])).array();
        fnump_int[q]=(*(fab_fnump_int[q])).array(); 
        fm[q] = fab_fm[q]->const_array();
        fp[q] = fab_fp[q]->const_array();
        dfm[q] = fab_dfm[q]->const_array();
        dfp[q] = fab_dfp[q]->const_array();
        um[q] = fab_um[q]->const_array();
        up[q] = fab_up[q]->const_array();
      }
            
      for(int q=0 ; q<Q; ++q){
        //compute the pointwise evaluations of the numerical flux
        amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
        {
          //check which indices it iterate across, i.e if last one is reachd
          fnum[q](i,j,k,m) = NumericalFlux(d,m,i,j,k,up[q],um[q],fp[q],fm[q],dfp[q],dfm[q]);  
        });          
        amrex::ParallelFor(bx, Np,[&] (int i, int j, int k, int n) noexcept
        {
            (fnumm_int[q])(i,j,k,n) = 0.0;
            (fnump_int[q])(i,j,k,n) = 0.0;        
        }); 
        for(int m = 0; m < M; ++m){ 
          amrex::ParallelFor(bx, Np,[&] (int i, int j, int k, int n) noexcept
          {
            (fnumm_int[q])(i,j,k,n)+=((fnum[q](i,j,k,m)*Mkbd[2*d][n][m])*dvol*dt);
            (fnump_int[q])(i,j,k,n)+=((fnum[q](i,j,k,m)*Mkbd[2*d+1][n][m])*dvol*dt);     
          });
        }
      }
    }
  }
}

amrex::Real AmrDG::PhysicalFlux(int lev, int d,int q, int m, int i, int j, int k,
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u,                             
                                amrex::Vector<amrex::Real> xi)
{
  //implementation of the physical flux present in the hyperbolic PDE
  amrex::Real f;
  f = sim->model_pde->pde_flux(lev,d,q,m,i,j,k,u,xi);
  return f; 
}

amrex::Real AmrDG::DPhysicalFlux(int lev, int d,int q, int m, int i, int j, int k,
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u,                             
                                amrex::Vector<amrex::Real> xi)
{
  //derivative of the physical flux of the hyperbolic PDE
  //if we are solving a system, then we have to specify here the unique eigenvalues of Jacobian of the flux
  amrex::Real df;
  df = sim->model_pde->pde_dflux(lev,d,q,m,i,j,k,u,xi);
  return df;
}

amrex::Real AmrDG::PhysicalSource(int lev,int q, int m, int i, int j, int k,
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                amrex::Vector<amrex::Real> xi)
{
  amrex::Real s;
  s =  sim->model_pde->pde_source(lev,q,m,i,j,k,u,xi);
  return s;
}

amrex::Real AmrDG::NumericalFlux(int d, int m,int i, int j, int k, 
                                amrex::Array4<const amrex::Real> up, 
                                amrex::Array4<const amrex::Real> um, 
                                amrex::Array4<const amrex::Real> fp,
                                amrex::Array4<const amrex::Real> fm,  
                                amrex::Array4<const amrex::Real> dfp,
                                amrex::Array4<const amrex::Real> dfm)
{
  //implementation of the numerical flux across interface
  //---------
  //    |        
  //  L | R      
  //    |
  //---------

  amrex::Real C;
  int shift[] = {0,0,0};
  amrex::Real uR,uL,fR,fL,DfR,DfL;

  shift[d] = -1;
  
  //L,R w.r.t boundary plus L==idx, R==idx+1
  uL  = up(i+shift[0],j+shift[1],k+shift[2],m);
  uR  = um(i,j,k,m);   
  fL  = fp(i+shift[0],j+shift[1],k+shift[2],m);
  fR  = fm(i,j,k,m);     
  DfL = dfp(i+shift[0],j+shift[1],k+shift[2],m);
  DfR = dfm(i,j,k,m);     
  C = (amrex::Real)std::max((amrex::Real)std::abs(DfL),(amrex::Real)std::abs(DfR));

  return 0.5*(fL+fR)-0.5*C*(uR-uL);
}

