#include "AmrDG.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

using namespace amrex;


AmrDG::L2ProjInterp::L2ProjInterp()
{
  interp_proj_mat();
}

void AmrDG::L2ProjInterp::interp_proj_mat()
{ 
  int num_overlap_cells = (int)std::pow(2,AMREX_SPACEDIM);

  //custom interpolation data structures
  amr_projmat_int.resize(num_overlap_cells,amrex::Vector<int >(AMREX_SPACEDIM));
  
  //coarse->fine projection matrix
  P_cf.resize(num_overlap_cells);
  for (int i = 0; i < num_overlap_cells; ++i) {
      P_cf[i].resize(numme->basefunc->Np_s, numme->basefunc->Np_s);
      P_cf[i].setZero();
  } 
  
  //fine->coarse projection matrix
  P_fc.resize((int)std::pow(2,AMREX_SPACEDIM));
  for (int i = 0; i < num_overlap_cells; ++i) {
      P_fc[i].resize(numme->basefunc->Np_s, numme->basefunc->Np_s);
      P_fc[i].setZero();
  } 

  //mass matrix (if multiplied by jacobian we get fine mass matrix, coarse mass matrix)
  //jacobian is simolified in formulation and only added when doing actual interpolation
  M.resize(numme->basefunc->Np_s,numme->basefunc->Np_s);
  M.setZero();

  //K=[-1,1]->amr->Km=[-1,0] u Kp=[0,1]
  //need all combinations of sub itnervalls that are obtained when refining. 
  //for simplificy just store the sing, i.e +,-
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

  //Compute mass matrices
  for(int j=0; j<numme->basefunc->Np_s;++j){
    for(int n=0; n<numme->basefunc->Np_s;++n){
      M(j,n)= numme->refMat_phiphi(j,numme->basefunc->basis_idx_s,n,numme->basefunc->basis_idx_s);
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singularValues = svd.singularValues();  
  Eigen::MatrixXd Minv = svd.matrixV() * singularValues.asDiagonal().inverse() 
                              * svd.matrixU().transpose();

  //Compute projection matrices for each sub-cell (indicated by idx)
  for(int l=0; l<std::pow(2,AMREX_SPACEDIM); ++l)
  { 
    //Define coordinate mapping between coarse cell and fine cell. 
    //depending on intervalls of fine cell in each dimension we need to shift differently
    //xi_f = 0.5*xi_c +-0.5 ==> "-":[-1,1]->[-1,0]  ,  "+":[-1,1]->[0,1]
    //since we store intervalls just as +1,-1 , we can directly use that value to get correct shift   
    //Construct mass matrix (identical do DG mass matrix, just cast it to Eigen)     
    amrex::Real shift[AMREX_SPACEDIM]={AMREX_D_DECL(0.5*amr_projmat_int[l][0],
                                        0.5*amr_projmat_int[l][1],
                                        0.5*amr_projmat_int[l][2])};

    for(int j=0; j<numme->basefunc->Np_s;++j){
      for(int m=0; m<numme->basefunc->Np_s;++m){

        //loop over quadrature points
        amrex::Real sum_cf = 0.0;
        amrex::Real sum_fc = 0.0;
        for(int q=0; q<numme->quadrule->qMp_s; ++q)
        {
          //Shift the quadrature point
          amrex::Real _xi_ref_shift[AMREX_SPACEDIM];
          for(int d=0; d<AMREX_SPACEDIM; ++d){
            _xi_ref_shift[d]=0.5*(numme->quadrule->xi_ref_quad_s[q][d])+shift[d];
          }  

          // Convert to const amrex::Vector<amrex::Real>
          const amrex::Vector<amrex::Real> xi_ref_shift(_xi_ref_shift, _xi_ref_shift + AMREX_SPACEDIM);
          
          //Use pre-computed QuadratureMatrix quadmat
          sum_cf += (numme->quadmat[j][q]*numme->basefunc->phi_s(m,numme->basefunc->basis_idx_s,xi_ref_shift));
          sum_fc += (numme->quadmat[m][q]*numme->basefunc->phi_s(j,numme->basefunc->basis_idx_s,xi_ref_shift));
        }

        P_cf[l](j, m)= sum_cf;

        P_fc[l](j, m)= sum_fc;

      }
    }
  }
}

Box AmrDG::L2ProjInterp::CoarseBox (const Box& fine,
                                    int        ratio)
{
    Box crse(amrex::coarsen(fine,ratio));
    //crse.grow(1);
    return crse; 
}

Box AmrDG::L2ProjInterp::CoarseBox (const Box&     fine,
                                    const IntVect& ratio)
{
    Box crse(amrex::coarsen(fine,ratio));
    //crse.grow(1);
    return crse; 
}

void AmrDG::L2ProjInterp::interp (const FArrayBox& crse,
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
            RunOn            runon)
{
  Array4<Real const> const& crsearr = crse.const_array();
  Array4<Real> const& finearr = fine.array();
  Box bx_c = crse.box();
  Box bx_c_ref = bx_c.refine(ratio);

  Box fb = fine.box()& fine_region;
  //fine.box(): returns the Box that represents the entire memory allocated for the FArrayBox named fine. 
  //This Box will include the valid region of fine and any ghost cells it might have.
  //creates the intersection of the FArrayBox's memory region (fine.box()) 
  //and the requested region to fill (fine_region). 
  //This ensures you only write to the target fine_region within the allocated fine array.

  //AMREX_PARALLEL_FOR_3D(runon,fb, i, j, k,
  amrex::ParallelFor(fb,[&] (int i, int j, int k) noexcept
  {  
    
    Eigen::VectorXd u_fine(ncomp);
    Eigen::VectorXd u_coarse(ncomp);

    //Loop over the components
    for(int n=0; n<ncomp;++n){
      u_fine[n] = finearr(i,j,k,fine_comp+n);
      u_coarse[n] =crsearr(i,j,k,crse_comp+n);
    }

    amr_scatter(i,j,k,u_fine,u_coarse, ratio);

    for(int n=0; n<ncomp;++n){
      finearr(i,j,k,fine_comp+n) = u_fine[n];
    }

  });  
}

void AmrDG::L2ProjInterp::amr_scatter(int i, int j, int k, Eigen::VectorXd& u_fine,
                                      Eigen::VectorXd& u_coarse, const amrex::IntVect& ratio) noexcept
{
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

  //Retrieve fine sub-cell idx
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

  u_fine = Minv*P_cf[idx]*u_coarse;

}
                                 
void AmrDG::L2ProjInterp::average_down(const MultiFab& S_fine, MultiFab& S_crse,
                int scomp, int ncomp, const IntVect& ratio, const int lev_fine, 
                const int lev_coarse) noexcept
{
  //TODO:placeholder code
}

void AmrDG::L2ProjInterp::amr_gather(int i, int j, int k, int n,Array4<Real> const& crse, 
                Array4<Real const> const& fine,int ccomp, 
                int fcomp, IntVect const& ratio) noexcept
{
  //TODO:placeholder code
}

/*
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
        }
        //else
        //{ 
        //  AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
        //  { 
        //    amr_gather_flux(i,j,k,n,d,crsearr,finearr,scomp,scomp,ratio);
        //  });         
        }
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
        //else
        //{
        //
        //  AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n,
        //  { 
        //    amr_gather_flux(i,j,k,n,d,crsearr,finearr,0,scomp,ratio);      
        //  });         
        //}
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




*/