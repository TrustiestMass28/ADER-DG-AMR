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
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_Interp_C.H>
#include <AMReX_MFInterp_C.H>

#include <climits>


void AmrDG::Limiter_w(int lev) 
{
  //NB: ghost cells cannot e limited, shouldnt be issue since expecitng 
  //disontinuity to be at center of refined lvl
  //  only approach to limit them would be to create mfabs grown by one extra 
  //ghost cell, exchange data and then limit those and copy values on normal mfab
  Print() <<"AmrDG::Limiter_w |lev="<<lev<<"\n";

  //this function takes in the modal representation of the DG polynomials and applies
  //limiting to the modes (_w) of all cells of level l
  
  //need to create MFab to store copy of U_w, this because we store there the 
  //limited/modifed modes
  //and then assing them to the actual U_w
  amrex::Vector<amrex::MultiFab> V_w(Q);
  amrex::Vector<amrex::MultiFab> tmp_U_p(Q);
  amrex::Vector<amrex::MultiFab> tmp_U_m(Q);
  
  for(int q=0 ; q<Q; ++q){
    //TODO:dependign on limiter used we need to initialize only certain modes
    //i.e depending on how many modes we need to modify and thus will require t
    //o be projected into characteristic field first
    if(limiter_type == "TVB")
    {
      V_w[q].define(U_w[lev][q].boxArray(), U_w[lev][q].DistributionMap(), AMREX_SPACEDIM, nghost);
      V_w[q].setVal(0.0);
      
      tmp_U_p[q].define(U_w[lev][q].boxArray(), U_w[lev][q].DistributionMap(), qMpbd, nghost);
      tmp_U_p[q].setVal(0.0);
      
      tmp_U_m[q].define(U_w[lev][q].boxArray(), U_w[lev][q].DistributionMap(), qMpbd, nghost);
      tmp_U_m[q].setVal(0.0);
    }
    else
    {
      V_w[q].define(U_w[lev][q].boxArray(), U_w[lev][q].DistributionMap(), Np, nghost);
      V_w[q].setVal(0.0);
    }    
  }
  
  amrex::Vector<amrex::MultiFab *> state_uw(Q);
  amrex::Vector< amrex::MultiFab *> state_vw(Q);
  amrex::Vector<amrex::MultiFab *> state_u(Q);
  amrex::Vector<amrex::MultiFab *> state_tmp_um(Q);
  amrex::Vector<amrex::MultiFab *> state_tmp_up(Q);
    
  for(int q=0; q<Q; ++q){
    state_tmp_um[q]=&(tmp_U_m[q]);
    state_tmp_up[q]=&(tmp_U_p[q]);
    state_uw[q]=&(U_w[lev][q]); 
    state_vw[q]=&(V_w[q]); 
    state_u[q]=&(U[lev][q]);
  }
  
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  { 
    
    amrex::Vector< amrex::FArrayBox *> fab_tmp_um(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > tmp_um(Q);
    amrex::Vector< amrex::FArrayBox *> fab_tmp_up(Q);
    amrex::Vector< amrex::Array4<  amrex::Real> > tmp_up(Q);
    amrex::Vector< amrex::FArrayBox *> fab_uw(Q);
    amrex::Vector< amrex::Array4< amrex::Real> > uw(Q); 
    amrex::Vector< amrex::FArrayBox *> fab_vw(Q);
    amrex::Vector< amrex::Array4<  amrex::Real> > vw(Q);   
    
    amrex::Vector< amrex::FArrayBox *> fab_u(Q);
    amrex::Vector< amrex::Array4<  amrex::Real>> u(Q);
    
    #ifdef AMREX_USE_OMP  
    for (MFIter mfi(*(state_uw[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)    
    #else
    for (MFIter mfi(*(state_uw[0]),true); mfi.isValid(); ++mfi)
    #endif 
    {
      const amrex::Box& bx = mfi.tilebox();  
      //const auto tagfab  = tags.array(mfi);
      for(int q=0 ; q<Q; ++q){
        fab_tmp_um[q] = &((*(state_tmp_um[q]))[mfi]);
        fab_tmp_up[q] = &((*(state_tmp_up[q]))[mfi]);
        fab_uw[q]=&((*(state_uw[q]))[mfi]);
        fab_vw[q]=&((*(state_vw[q]))[mfi]);
        fab_u[q]=&((*(state_u[q]))[mfi]);
        
        tmp_um[q] = (*(fab_tmp_um[q])).array();
        tmp_up[q] = (*(fab_tmp_up[q])).array();
        uw[q]=(*(fab_uw[q])).array();
        vw[q]=(*(fab_vw[q])).array();
        u[q]=(*(fab_u[q])).array();
      }
  
      amrex::Dim3 lo = lbound(bx);
      amrex::Dim3 hi = ubound(bx);
      amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
      {
        if(limiter_type == "TVB")
        {
          Limiter_linear_tvb(i, j,  k, &uw, &tmp_um, &tmp_um, &vw, lev);
        }
 
        //get limited values of derived quantities 
        if(Q_unique != Q){
          //currently work only for derived quantities defined in cell volume
          //derived qty defined on cell volume
          int M = qMp_L2proj; //only spatial points
          for(int m = 0; m<M ; ++m){
            // need just pointwise evaluation of the unique quantities, while now its 
            //going to compute also for derived one, bit of a waste
            get_u_from_u_w (m, i, j, k,&uw, &u, xi_ref_GLquad_L2proj[m]);//xi[m]
   
            for(int q=Q_unique; q<Q; ++q){
              //computed limited pointwise evaluation of derived quantites, 
              //overwritten in correct location of u
              sim->model_pde->pde_derived_qty(lev,q,m,i,j,k,&u,xi_ref_GLquad_L2proj[m]);      
            }
          }
          
          //get modal representation of limited derived quantities
          for(int q=Q_unique; q<Q; ++q){
            for(int n = 0; n<Np ; ++n){
              amrex::Real sum = 0.0;
              for(int m=0; m<qMp_L2proj; ++m)
              {
                sum+= ((u)[q])(i,j,k,m)*L2proj_quadmat[n][m];              
              }
              ((uw)[q])(i,j,k,n) = (sum/(RefMat_phiphi(n,n, false, false)));
            }
          } 
        }
        
      });
    }    
  } 
  
  //sync internal ghost, cant do anything about external ones/interface fine coarse
  for(int q=0; q<Q; ++q){
    U_w[lev][q].FillBoundary(geom[lev].periodicity()); 
  }
}

void AmrDG::Limiter_linear_tvb(int i, int j, int k, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* uw, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* um,
                              amrex::Vector<amrex::Array4<amrex::Real>>* up, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* vw,
                              int lev)
{
  amrex::Vector<amrex::Vector<amrex::Real>> res_limit(AMREX_SPACEDIM, 
                                              amrex::Vector<amrex::Real>(Q_unique, 0.0));
                                              
  amrex::Vector<amrex::Vector<amrex::Real>> L_EV;
  amrex::Vector<amrex::Vector<amrex::Real>> R_EV;
  
  amrex::Vector<int> troubled_flag(Q_unique,0);
  
  //loop over dimensions and for each one select the correct linear basis function idx
  for(int lin_idx=0; lin_idx<AMREX_SPACEDIM; ++lin_idx){
    //get idx of linear modes
    int s = lin_mode_idx[lin_idx];
    int shift[] = {0,0,0};
    int _d;
    
    //we know that basis function s is linear, need to find in which direction _d is linear
    //so that we can shift cell indices correctly
    for(int d=0; d<AMREX_SPACEDIM; ++d)
    {
      if(mat_idx_s[s][d] == 1)
      {
        shift[d] = 1; 
        _d = d;
        break;
      }
    }       

    amrex::Real Dm_u_avg;
    amrex::Real Dp_u_avg;
    amrex::Real D_u;

    //Get left and right eiugenvalue matrix, we use cell average to compute it
    L_EV = sim->model_pde->pde_EV_Lmatrix(_d,0,i,j,k,uw);
    
    //get decoupled characteristic field variables

    for(int q=0; q<Q_unique; ++q){
    
      Dm_u_avg = 0.0;
      Dp_u_avg = 0.0;
      D_u = 0.0;
      
      
      for(int _q=0; _q<Q_unique; ++_q)
      {
        Dm_u_avg += 0.5*L_EV[q][_q]*(((*uw)[_q])(i,j,k,0)-
                                          ((*uw)[_q])(i-shift[0],j-shift[1],k-shift[2],0));
                                          
        Dp_u_avg += 0.5*L_EV[q][_q]*(((*uw)[_q])(i+shift[0],j+shift[1],k+shift[2],0)
                                          -((*uw)[_q])(i,j,k,0));        
                                                  
        D_u +=L_EV[q][_q]*((*uw)[_q])(i,j,k,s);            
      }
      bool tmp_flag =false;
      
      (*vw)[q](i,j,k,lin_idx)  = minmodB(D_u,Dm_u_avg,Dp_u_avg, tmp_flag, lev);    

      //if limiting required along one direction, it will be applied to all directions
      //therefore just need one flag per component to represent if we need limiting
      //when its needed along a dimension we store flag as true, using or (||) 
      //operator we make sure that if at least in one direction it is true, the final 
      //component flag is true
      //AllPrint() <<troubled_flag[q] <<"\n";
      troubled_flag[q] = (troubled_flag[q] || tmp_flag);
    }
    
    R_EV = sim->model_pde->pde_EV_Rmatrix(_d,0,i,j,k,uw);
    //now need to convert back to conservative modes
    for (int q = 0; q < Q_unique; ++q){
      amrex::Real sum=0.0;
      for (int _q = 0; _q < Q_unique; ++_q){
        sum+=R_EV[q][_q]*((*vw)[_q])(i,j,k,lin_idx);
      }
      res_limit[lin_idx][q]=sum;         
    }
  }
 
  for (int q = 0; q < Q_unique; ++q){
    if(troubled_flag[q]){//if limiting required in one direction, doit in all
      
      for(int n=1; n<Np; ++n){
        (*uw)[q](i,j,k,n) = 0.0;
      }

      for(int lin_idx=0; lin_idx<AMREX_SPACEDIM; ++lin_idx){  
        int s = lin_mode_idx[lin_idx];   
        (*uw)[q](i,j,k,s)= res_limit[lin_idx][q];
      } 
    }
  }    
  
}

void AmrDG::get_u_from_u_w(int c, int i, int j, int k,
                          amrex::Vector<amrex::Array4< amrex::Real>>* uw, 
                          amrex::Vector<amrex::Array4< amrex::Real>>* u ,
                          amrex::Vector<amrex::Real> xi)
{
  //computes the sum of modes and respective basis function evaluated at specified location
  //for all solution components
  for(int q=0 ; q<Q; ++q){
    amrex::Real sum = 0.0;
    for (int n = 0; n < Np; ++n){  
      sum+=(((*uw)[q])(i,j,k,n)*Phi(n, xi));
    }
    ((*u)[q])(i,j,k,c) = sum;
  }
}

amrex::Real AmrDG::minmodB(amrex::Real a1,amrex::Real a2,amrex::Real a3, 
                          bool &troubled_flag, int l) const
{
  amrex::Real h;  
  auto const dx = geom[l].CellSizeArray();
  h = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});

  if(std::abs(a1)<= AMR_TVB_C[l]*TVB_M*std::pow(h,2.0))
  {    
    troubled_flag = false; 
    return a1;
  }
  else
  {
    troubled_flag = true;
    return minmod(a1,a2,a3, troubled_flag);
  }
}

amrex::Real AmrDG::minmod(amrex::Real a1,amrex::Real a2,amrex::Real a3, 
                          bool &troubled_flag) const
{
  bool sameSign = (std::signbit(a1) == std::signbit(a2)) &&
                (std::signbit(a2) == std::signbit(a3));
  int sign;
  if(std::signbit(a1) == std::signbit(-1))
  {sign = -1;}
  else
  {sign = +1;}
   
  if(sameSign)
  {
    troubled_flag = true;
    return sign*std::min({std::abs(a1), std::abs(a2), std::abs(a3)});
  }
  else
  {
    troubled_flag = false; 
    return 0;
  }
}




















