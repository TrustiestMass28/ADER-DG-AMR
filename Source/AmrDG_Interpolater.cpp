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
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

/*
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
*/
AmrDG::DGprojInterp custom_interp;
//extern AMREX_EXPORT AmrDG::DGprojInterp custom_interp;
void AmrDG::DGprojInterp::getouterref(AmrDG* _amrdg)
{
  amrdg= _amrdg;
}

void AmrDG::DGprojInterp::interp_proj_mat()
{
  //custom interpolation data structures
  amr_projmat_int.resize(std::pow(2,AMREX_SPACEDIM),amrex::Vector<int >(AMREX_SPACEDIM));
  
  P_scatter.resize((int)std::pow(2,AMREX_SPACEDIM),
            amrex::Vector<amrex::Vector<amrex::Real>>(amrdg->Np,
            amrex::Vector<amrex::Real>(amrdg->Np)));
            
  P_gather.resize((int)std::pow(2,AMREX_SPACEDIM),
            amrex::Vector<amrex::Vector<amrex::Real>>(amrdg->Np,
            amrex::Vector<amrex::Real>(amrdg->Np)));
  
  //K=[-1,1]->amr->Km=[-1,0] u Kp=[0,1]
  //need all combinations of sub itnervalls that are obtained when refining. 
 
  amrex::Vector<int> Kpm_int = {-1,1}; //=={Km,Kp}
  //NB: the product of the plus [0,1] minus [-1,0] intervalls defined in each 
  //entry of amr_projmat_int[idx] define completely
  //a finer cell. e.g amr_projmat_int[idx] = [-1,1,-1] ==> [-1,0]x[0,1]x[-1,0]

  #if (AMREX_SPACEDIM == 1)
  for(int i=0; i<2;++i){
    amr_projmat_int[i][0]= Kpm_int[i];
  }   
  #elif (AMREX_SPACEDIM == 2)
  for(int i=0; i<2;++i){
    for(int j=0; j<2;++j){
      amr_projmat_int[2*i+j][0]= Kpm_int[i];
      amr_projmat_int[2*i+j][1]= Kpm_int[j];
    }
  }  
  #elif (AMREX_SPACEDIM == 3)
  for(int i=0; i<2;++i){
    for(int j=0; j<2;++j){
      for(int k=0; k<2;++k){
        amr_projmat_int[2*2*i+2*j+k][0]= Kpm_int[i];
        amr_projmat_int[2*2*i+2*j+k][1]= Kpm_int[j];
        amr_projmat_int[2*2*i+2*j+k][2]= Kpm_int[k];
      }
    }
  }  
  #endif
  
  for(int idx=0; idx<std::pow(2,AMREX_SPACEDIM); ++idx)
  { 
    //Define coordinate mapping between coarse cell and fine cell. 
    //depending on intervalls of fine cell in each dimension we need to shift differently
    //xi_f = 0.5*xi_c +-0.5 ==> "-":[-1,1]->[-1,0]  ,  "+":[-1,1]->[0,1]
    //since we store intervalls just as +1,-1 , we can directly use that value to get correct shift   
    //Construct mass matrix (identical do DG mass matrix, just cast it to Eigen)     
    amrex::Real shift[AMREX_SPACEDIM]={AMREX_D_DECL(0.5*amr_projmat_int[idx][0],
                                        0.5*amr_projmat_int[idx][1],
                                        0.5*amr_projmat_int[idx][2])};

    //Mass matrix
    Eigen::MatrixXd M(amrdg->Np, amrdg->Np);
    for (int i = 0; i < amrdg->Np; ++i){
      for (int j = 0; j < amrdg->Np; ++j){
        amrex::Real val = 1.0;
        for(int d=0; d<AMREX_SPACEDIM; ++d){
          val*=(amrex::Real)amrdg->KroneckerDelta(amrdg->mat_idx_s[i][d],amrdg->mat_idx_s[j][d])
              *(2.0/(2.0*(amrex::Real)amrdg->mat_idx_s[i][d]+1.0));
        }
        M(i,j)=val; 
      }
    }
    
    //Stiffness matrix
    Eigen::MatrixXd S(amrdg->Np, amrdg->Np);
    for (int i = 0; i < amrdg->Np; ++i) {
      for (int j = 0; j < amrdg->Np; ++j) {
        amrex::Real sum = 0.0;
        for(int q=0; q<amrdg->qMp_L2proj; ++q)
        {
          amrex::Real xi_ref_GLquad_L2proj_mod[AMREX_SPACEDIM];
          for(int d=0; d<AMREX_SPACEDIM; ++d){
            xi_ref_GLquad_L2proj_mod[d]=0.5*(amrdg->xi_ref_GLquad_L2proj[q][d])+shift[d];
          }  
          
          amrex::Real phi_shift = 1.0;
          for  (int d = 0; d < AMREX_SPACEDIM; ++d){
            phi_shift*=std::legendre(amrdg->mat_idx_s[j][d], xi_ref_GLquad_L2proj_mod[d]);
          }

          //compute quadrature weight
          amrex::Real w=1.0;
          int N = amrdg->qMp_1d; 
          for(int d=0; d<AMREX_SPACEDIM; ++d){
            w*=2.0/std::pow(std::assoc_legendre(N,1,amrdg->xi_ref_GLquad_L2proj[q][d]),2.0);
          }
          
          amrex::Real phi = 1.0;
          for  (int d = 0; d < AMREX_SPACEDIM; ++d){
            phi*=std::legendre(amrdg->mat_idx_s[i][d], amrdg->xi_ref_GLquad_L2proj[q][d]);
          }           
          
          sum+=(phi_shift*phi*w); 
        }
        S(i,j)=sum;
      }
    }
    
    Eigen::MatrixXd Ps(amrdg->Np, amrdg->Np);
    Ps= (M.inverse())*S;    
    for (int i = 0; i < amrdg->Np; ++i) {
      for (int j = 0; j < amrdg->Np; ++j) {
        P_scatter[idx][i][j] =  (amrex::Real)Ps(i,j);    
      }
    }
    
    Eigen::MatrixXd Pg(amrdg->Np, amrdg->Np);
    Pg = ((M.inverse())*(S.transpose()))*(1.0/std::pow(2.0,AMREX_SPACEDIM));//
    for (int i = 0; i < amrdg->Np; ++i) {
      for (int j = 0; j < amrdg->Np; ++j) {
        P_gather[idx][i][j] = (amrex::Real)Pg(i,j); 
      }
    }
  }

}

amrex::Box AmrDG::DGprojInterp::CoarseBox (const Box& fine, int ratio)
{  
  Box crse(amrex::coarsen(fine,ratio));
  //crse.grow(1);
  return crse; 

}

amrex::Box AmrDG::DGprojInterp::CoarseBox (const Box& fine, const IntVect& ratio)
{ 
  Box crse(amrex::coarsen(fine,ratio));
  //crse.grow(1);
  return crse; 
}

void AmrDG::DGprojInterp::interp (const FArrayBox& crse, int  crse_comp, FArrayBox&  fine, 
                                  int  fine_comp,int   ncomp,const Box&  fine_region, 
                                  const IntVect&  ratio, const Geometry& crse_geom, 
                                  const Geometry& fine_geom,  Vector<BCRec> const& bcr,
                                  int actual_comp, int  actual_state,RunOn runon)
{
  Array4<Real const> const& crsearr = crse.const_array();
  Array4<Real> const& finearr = fine.array();
  Box bx_c = crse.box();
  Box bx_c_ref = bx_c.refine(ratio);

  Box fb = fine.box()& fine_region;

  AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FLAG(runon,fb, ncomp, i, j, k, n,
  {  
    amr_scatter(i,j,k,n, finearr, fine_comp, crsearr, crse_comp, ncomp, ratio);
  });  
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void AmrDG::DGprojInterp::amr_scatter(int i, int j, int k, int n, Array4<Real> const& fine,
                                      int fcomp,  Array4<Real const> const& crse, int ccomp, 
                                      int ncomp, IntVect const& ratio) noexcept
{
  //mesh loop is over fine indices, we anyway compute our own fine idx (again) 
  //so that we can then comapre them and select 
  //appropriate idx of projection matrix  
#if (AMREX_SPACEDIM == 1)
  int i_c = amrex::coarsen(i,ratio[0]);
  int j_c = 0;
  int k_c = 0;
#elif (AMREX_SPACEDIM == 2)
  int i_c = amrex::coarsen(i,ratio[0]);
  int j_c = amrex::coarsen(j,ratio[1]);
  int k_c = 0;
#elif (AMREX_SPACEDIM == 3)  
  int i_c = amrex::coarsen(i,ratio[0]);
  int j_c = amrex::coarsen(j,ratio[1]);
  int k_c = amrex::coarsen(j,ratio[2]);
#endif

  //Retrieve idx of subcell given fine and coarse idx relation
  int si,sj,sk;
#if (AMREX_SPACEDIM == 1)
  if(i_c<0 && i == i_c){si=1;}
  else if(i_c>=0 && i == 2*i_c){si=-1;}
  else if(i_c>=0 && i == 2*i_c+1){si=1;}
#elif (AMREX_SPACEDIM == 2)
  if(i_c<0){si=1;}
  else if(i_c>=0 && i == 2*i_c){si=-1;}
  else if(i_c>=0 && i == 2*i_c+1){si=1;}
  
  if(j_c<0){sj=1;}
  else if(j_c>=0 && j == 2*j_c){sj=-1;}
  else if(j_c>=0 && j == 2*j_c+1){sj=1;}  
#elif (AMREX_SPACEDIM == 3) 
  if(i_c<0 && i == i_c){si=1;}
  else if(i_c>=0 && i == 2*i_c){si=-1;}
  else if(i_c>=0 && i == 2*i_c+1){si=1;}
  
  if(j_c<0 && j == j_c){sj=1;}
  else if(j_c>=0 && j == 2*j_c){sj=-1;}
  else if(j_c>=0 && j == 2*j_c+1){sj=1;}  
  
  if(k_c<0 && k == k_c){sk=1;}
  else if(k_c>=0 && k == 2*k_c){sk=-1;}
  else if(k_c>=0 && k == 2*k_c+1){sk=1;}
#endif

  
  int idx;
  for(int _idx=0; _idx<std::pow(2,AMREX_SPACEDIM); ++_idx)
  {
  #if (AMREX_SPACEDIM == 1)
    if((si ==amr_projmat_int[_idx][0]))
  #elif (AMREX_SPACEDIM == 2)
    if((si ==amr_projmat_int[_idx][0]) && (sj ==amr_projmat_int[_idx][1]))
  #elif (AMREX_SPACEDIM == 3) 
    if((si ==amr_projmat_int[_idx][0]) && (sj ==amr_projmat_int[_idx][1])
        && (sj ==amr_projmat_int[_idx][2]))
  #endif
    {
      idx=_idx;
      break;
    }
  }

  //Scatter data from coarse to fine
  amrex::Real sum=0.0;
  for(int m=0; m<(amrdg->Np);++m){
    sum+= (P_scatter[idx][n][m]*crse(i_c,j_c,k_c,m));
  } 
  
  fine(i,j,k,n)= sum; 
}

void AmrDG::DGprojInterp::average_down(const MultiFab& S_fine, MultiFab& S_crse,
                                      int scomp, int ncomp, const IntVect& ratio, 
                                      const int lev_fine, const int lev_coarse, 
                                      int d, bool flag_flux)
{
  //average down not! applied to ghost cells. Only to valid cells
  // Coarsen() the fine stuff on processors owning the fine data.
  const BoxArray& fine_BA = S_fine.boxArray();
  const DistributionMapping& fine_dm = S_fine.DistributionMap();
  BoxArray crse_S_fine_BA = fine_BA;
  crse_S_fine_BA.coarsen(ratio);
  
  
  if (crse_S_fine_BA == S_crse.boxArray() && S_fine.DistributionMap() == S_crse.DistributionMap())
  {     
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
      for (MFIter mfi(S_crse,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        //  NOTE: The tilebox is defined at the coarse level.
        const Box& bx = mfi.tilebox();
        Array4<amrex::Real> const& crsearr = S_crse.array(mfi);
        Array4<amrex::Real const> const& finearr = S_fine.const_array(mfi);
        
        if(!flag_flux){
          AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
          {
            amr_gather(i,j,k,n,crsearr,finearr,scomp,scomp,ratio);
          });    
        }/*
        else
        { 
          AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
          { 
            amr_gather_flux(i,j,k,n,d,crsearr,finearr,scomp,scomp,ratio);
          });         
        }*/
      }
    }
  }    
  else
  {
    MultiFab crse_S_fine(crse_S_fine_BA,fine_dm,ncomp,0,MFInfo(),FArrayBoxFactory());
      
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
      for (MFIter mfi(crse_S_fine,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        //  NOTE: The tilebox is defined at the coarse level.
        const Box& bx = mfi.tilebox();
        Array4<amrex::Real> const& crsearr = crse_S_fine.array(mfi);
        Array4<amrex::Real const> const& finearr = S_fine.const_array(mfi);

        //  NOTE: We copy from component scomp of the fine fab into component 0 of the crse fab
        //        because the crse fab is a temporary which was made starting at comp 0, it is
        //        not part of the actual crse multifab which came in.
        if(!flag_flux){
          AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
          {
            amr_gather(i,j,k,n,crsearr,finearr,0,scomp,ratio);      
          });  
        }/*
        else
        {

          AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
          { 
            amr_gather_flux(i,j,k,n,d,crsearr,finearr,0,scomp,ratio);      
          });         
        }*/
      }
    }

    S_crse.ParallelCopy(crse_S_fine,0,scomp,ncomp);
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void AmrDG::DGprojInterp::amr_gather(int i, int j, int k, int n,Array4<Real> const& crse, 
                                      Array4<Real const> const& fine,int ccomp, 
                                      int fcomp, IntVect const& ratio) noexcept
{

  //reset coarse first 
  crse(i,j,k,n+ccomp) = 0.0;
  
#if (AMREX_SPACEDIM == 1)
  const int faci = ratio[0];
  
  const int ii = i*faci;
  
  int i_f,j_f,k_f;
  for (int iref = 0; iref < faci; ++iref){        
    i_f=ii+iref;
    j_f = 0;
    k_f = 0;

    int si = (i_f % faci == 0) ? -1 : 1;
            
    int idx;
    for(int _idx=0; _idx<std::pow(2,AMREX_SPACEDIM); ++_idx)
    { 
      if((si ==amr_projmat_int[_idx][0]))
      {
        idx=_idx;
        break;
      }
    }
    
    amrex::Real sum=0.0;
    for(int m=0; m<(amrdg->Np);++m){
      sum+= (P_gather[idx][n][m]*fine(i_f,j_f,k_f,m));
    } 
    crse(i,j,k,n)+=sum;
  }
   
#elif (AMREX_SPACEDIM == 2) 
  const int faci = ratio[0];
  const int facj = ratio[1];
  
  const int ii = i*faci;
  const int jj = j*facj;
  
  int i_f,j_f,k_f; 
  for (int jref = 0; jref < facj; ++jref){
    for (int iref = 0; iref < faci; ++iref){   
      i_f=ii+iref;
      j_f=jj+jref;
      k_f = 0;
 
      int si = (i_f % faci == 0) ? -1 : 1;
      int sj = (j_f % facj == 0) ? -1 : 1;
              
      int idx;
      for(int _idx=0; _idx<std::pow(2,AMREX_SPACEDIM); ++_idx)
      { 
        if((si ==amr_projmat_int[_idx][0]) && (sj ==amr_projmat_int[_idx][1]))
        {
          idx=_idx;
          break;
        }
      }

      amrex::Real sum=0.0;
      for(int m=0; m<(amrdg->Np);++m){
        sum+= (P_gather[idx][n][m]*fine(i_f,j_f,k_f,m));
      } 
      crse(i,j,k,n)+=sum;
    }
  } 
#elif (AMREX_SPACEDIM == 3)
  const int faci = ratio[0];
  const int facj = ratio[1];
  const int fack = ratio[2];

  const int ii = i*faci;
  const int jj = j*facj;
  const int kk = k*fack;
  
  int i_f,j_f,k_f;
  for (int kref = 0; kref < fack; ++kref){
    for (int jref = 0; jref < facj; ++jref){
      for (int iref = 0; iref < faci; ++iref){        
        i_f=ii+iref;
        j_f=jj+jref;
        k_f=kk+kref;
   
        int si = (i_f % faci == 0) ? -1 : 1;
        int sj = (j_f % facj == 0) ? -1 : 1;
        int sk = (k_f % fack == 0) ? -1 : 1;
                
        int idx;
        for(int _idx=0; _idx<std::pow(2,AMREX_SPACEDIM); ++_idx)
        { 
          if((si ==amr_projmat_int[_idx][0]) && (sj ==amr_projmat_int[_idx][1]) 
            && (sk ==amr_projmat_int[_idx][2]))
          {
            idx=_idx;
            break;
          }
        }
        
        amrex::Real sum=0.0;
        for(int m=0; m<(amrdg->Np);++m){
          sum+= (P_gather[idx][n][m]*fine(i_f,j_f,k_f,m));
        } 
        crse(i,j,k,n)+=sum;
      }
    }
  }  
#endif  
}

void AmrDG::DGprojInterp::average_down_flux(MultiFab& S_fine, MultiFab& S_crse,
                                      int scomp, int ncomp, const IntVect& ratio, 
                                      const int lev_fine, const int lev_coarse, 
                                      int d, bool flag_flux)
{
  //average down not! applied to ghost cells. Only to valid cells
  // Coarsen() the fine stuff on processors owning the fine data.
  const BoxArray& fine_BA = S_fine.boxArray();
  const DistributionMapping& fine_dm = S_fine.DistributionMap();
  BoxArray crse_S_fine_BA = fine_BA;
  crse_S_fine_BA.coarsen(ratio);
  
  
  if (crse_S_fine_BA == S_crse.boxArray() && S_fine.DistributionMap() == S_crse.DistributionMap())
  {     
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
      for (MFIter mfi(S_crse,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        //  NOTE: The tilebox is defined at the coarse level.
        const Box& bx = mfi.tilebox();
        Array4<amrex::Real> const& crsearr = S_crse.array(mfi);
        Array4<amrex::Real > const& finearr = S_fine.array(mfi);
        

          AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
          { 
            amr_gather_flux(i,j,k,n,d,crsearr,finearr,scomp,scomp,ratio);
          });         
        
      }
    }
  }    
  else
  {
    MultiFab crse_S_fine(crse_S_fine_BA,fine_dm,ncomp,0,MFInfo(),FArrayBoxFactory());
      
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
      for (MFIter mfi(crse_S_fine,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        //  NOTE: The tilebox is defined at the coarse level.
        const Box& bx = mfi.tilebox();
        Array4<amrex::Real> const& crsearr = crse_S_fine.array(mfi);
        Array4<amrex::Real> const& finearr = S_fine.array(mfi);

        //  NOTE: We copy from component scomp of the fine fab into component 0 of the crse fab
        //        because the crse fab is a temporary which was made starting at comp 0, it is
        //        not part of the actual crse multifab which came in.


          AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
          { 
            amr_gather_flux(i,j,k,n,d,crsearr,finearr,0,scomp,ratio);      
          });         
        
      }
    }

    S_crse.ParallelCopy(crse_S_fine,0,scomp,ncomp);
  }
}

/*
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void AmrDG::DGprojInterp::amr_gather_flux(int i, int j, int k, int n, int d,Array4<Real> const& crse, 
                                          Array4<Real const> const& fine,int ccomp, 
                                          int fcomp, IntVect const& ratio) noexcept
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void AmrDG::DGprojInterp::amr_gather_flux(int i, int j, int k, int n, int d,Array4<Real> const& crse, 
                                          Array4<Real> const& fine,int ccomp, 
                                          int fcomp, IntVect const& ratio) noexcept
{
  //TODO: should skip coarse indices that are not at fine coarse interface
  //  get boxArray max and min indices and compare i,j,k
  //should be dimension dependent, i.e if d==0 then take all j and msut be min(i) or max(i)
  //can be done before function call

#if (AMREX_SPACEDIM == 1)  
  const int faci = ratio[0]; 
  const int ii = i*faci;

  int i_f,j_f,k_f; 
  
  if(d==0)
  {
    crse(i,j,k,n+ccomp) = 0.0;
      i_f=ii;
      j_f= 0;
      k_f = 0;
      crse(i,j,k,n+ccomp)+=(fine(i_f,j_f,k_f,n+fcomp));
  }

#elif (AMREX_SPACEDIM == 2)   
  const int faci = ratio[0];
  const int facj = ratio[1];
  
  const int ii = i*faci;
  const int jj = j*facj;
  
  int i_f,j_f,k_f; 
  
  if(d==0)
  {
    crse(i,j,k,n+ccomp) = 0.0;
    for (int jref = 0; jref < facj; ++jref){
      i_f=ii;
      j_f=jj+jref;
      k_f = 0;
      crse(i,j,k,n+ccomp)+=(fine(i_f,j_f,k_f,n+fcomp));
    }
  }
  else if(d==1)
  {
    crse(i,j,k,n+ccomp) = 0.0;
    for (int iref = 0; iref < faci; ++iref){
      i_f = ii+iref;
      j_f = jj;
      k_f =0;
      
      crse(i,j,k,n+ccomp)+=(fine(i_f,j_f,k_f,n+fcomp));
    }    
  }  
#elif (AMREX_SPACEDIM == 3) 
  const int faci = ratio[0];
  const int facj = ratio[1];
  const int fack = ratio[2];
  
  const int ii = i*faci;
  const int jj = j*facj;
  const int kk = k*fack;
  
  int i_f,j_f,k_f; 
  
  if(d==0)
  {
    crse(i,j,k,n+ccomp) = 0.0;
    for (int kref = 0; kref < fack; ++kref){
      for (int jref = 0; jref < facj; ++jref){
        i_f=ii;
        j_f=jj+jref;
        k_f = kk+kref;
        crse(i,j,k,n+ccomp)+=(fine(i_f,j_f,k_f,n+fcomp));
      }
    }
  }
  else if(d==1)
  {
    crse(i,j,k,n+ccomp) = 0.0;
    for (int kref = 0; kref < fack; ++kref){
      for (int iref = 0; iref < faci; ++iref){
        i_f = ii+iref;
        j_f = jj;
        k_f =kk+kref;
        
        crse(i,j,k,n+ccomp)+=(fine(i_f,j_f,k_f,n+fcomp));
      }  
    }
  }
  else if(d==2)
  {
    crse(i,j,k,n+ccomp) = 0.0;
    for (int jref = 0; jref < facj; ++jref){
      for (int iref = 0; iref < faci; ++iref){
        i_f = ii+iref;
        j_f = jj+jref;
        k_f =kk;
        
        crse(i,j,k,n+ccomp)+=(fine(i_f,j_f,k_f,n+fcomp));
      }  
    }
  } 
#endif  

}

