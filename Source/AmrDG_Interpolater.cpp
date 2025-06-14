#include "AmrDG.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

using namespace amrex;

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
  Minv = svd.matrixV() * singularValues.asDiagonal().inverse() 
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
  Print() <<"INTERPOLATING FROM COARSE"<<"\n";
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
    amr_scatter(i,j,k,finearr,fine_comp,crsearr,crse_comp,ncomp,ratio);
  });  
}

void AmrDG::L2ProjInterp::average_down(const MultiFab& S_fine, int fine_comp, MultiFab& S_crse, 
                                      int crse_comp, int ncomp, const IntVect& ratio, 
                                      const int lev_fine, const int lev_coarse) noexcept
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
        
        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {  
          amr_gather(i,j,k,finearr,fine_comp,crsearr,crse_comp,ncomp,ratio);
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
        Array4<amrex::Real const> const& finearr = S_fine.const_array(mfi);

        //  NOTE: We copy from component scomp of the fine fab into component 0 of the crse fab
        //        because the crse fab is a temporary which was made starting at comp 0, it is
        //        not part of the actual crse multifab which came in.

        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {  
          amr_gather(i,j,k,finearr,fine_comp,crsearr,crse_comp,ncomp,ratio);
        });  
      }
    }
    S_crse.ParallelCopy(crse_S_fine,0,fine_comp,ncomp);
  }
}

void AmrDG::L2ProjInterp::amr_scatter(int i, int j, int k, Array4<Real> const& fine, 
                                    int fcomp, Array4<Real const> const& crse, int ccomp, 
                                    int ncomp, IntVect const& ratio) noexcept
{   
    
    Eigen::VectorXd u_fine(ncomp);
    Eigen::VectorXd u_coarse(ncomp);

    auto map = set_fine_coarse_idx_map(i,j,k,ratio);

    //Loop over the components
    for(int n=0; n<ncomp;++n){
      u_coarse[n] =crse(map.i,map.j,map.k,ccomp+n);
    }

    u_fine = Minv*P_cf[map.fidx]*u_coarse;

    for(int n=0; n<ncomp;++n){
      fine(i,j,k,fcomp+n) = u_fine[n];
    }
}

void AmrDG::L2ProjInterp::amr_gather(int i, int j, int k,  Array4<Real const> const& fine,int fcomp,
                                     Array4<Real> const& crse, int ccomp, int ncomp, 
                                     IntVect const& ratio ) noexcept
             
{

  int num_overlap_cells = (int)std::pow(2,AMREX_SPACEDIM);

  amrex::Vector<Eigen::VectorXd> u_fine;
  u_fine.resize(num_overlap_cells,Eigen::VectorXd (ncomp));

  Eigen::VectorXd u_coarse(ncomp);
  u_coarse.setZero();

  auto map = set_coarse_fine_idx_map(i,j,k,ratio);

  Eigen::VectorXd sum(ncomp);
  for(int l=0; l<num_overlap_cells;++l)
  {
    for(int n=0; n<ncomp;++n){
      u_fine[l][n] = fine(map[l].i,map[l].j,map[l].k,fcomp+n);
    }    
    sum+=P_fc[map[l].fidx]*u_fine[l];
  }
  
  u_coarse = (1.0/num_overlap_cells)*Minv*sum;

  for(int n=0; n<ncomp;++n){
    crse(i,j,k,ccomp+n) = u_coarse[n];
  }
}

AmrDG::L2ProjInterp::IndexMap AmrDG::L2ProjInterp::set_fine_coarse_idx_map(int i, int j, int k, const amrex::IntVect& ratio)
{
    //pass fine cell index and return overlapping coarse cell index
    //and index locating fine cell w.r.t coarse one reference frame
    IndexMap _map;

    // --- Calculate Coarse Cell Indices ---
#if (AMREX_SPACEDIM == 1)
    int _i = amrex::coarsen(i,ratio[0]);
    int _j = 0; // Not used in 1D
    int _k = 0; // Not used in 1D
#elif (AMREX_SPACEDIM == 2)
    int _i = amrex::coarsen(i,ratio[0]);
    int _j = amrex::coarsen(j,ratio[1]);
    int _k = 0; // Not used in 2D
#elif (AMREX_SPACEDIM == 3)
    int _i = amrex::coarsen(i,ratio[0]);
    int _j = amrex::coarsen(j,ratio[1]);
    int _k = amrex::coarsen(k,ratio[2]);
#endif

    // Populate coarse cell indices in _map
    _map.i = _i;
    _map.j = _j;
    _map.k = _k;

    // --- Determine Sub-Cell Relative Position (si, sj, sk) ---
    // These values indicate if the fine cell is in the "lower" (-1) or "upper" (1) half
    // of the coarse cell along each dimension.
    // This logic assumes a refinement ratio of 2.
    int si, sj, sk;

#if (AMREX_SPACEDIM == 1)
    // For 1D, if fine index 'i' is even, it's the lower half; if odd, it's the upper half.
    // This assumes ratio[0] is 2.
    if ((i - _i * ratio[0]) == 0) { // Equivalent to i % ratio[0] == 0 for fine_idx >= 0
        si = -1;
    } else { // (i - _i * ratio[0]) == 1
        si = 1;
    }

#elif (AMREX_SPACEDIM == 2)
    // For x-dimension
    if ((i - _i * ratio[0]) == 0) {
        si = -1;
    } else {
        si = 1;
    }
    // For y-dimension
    if ((j - _j * ratio[1]) == 0) {
        sj = -1;
    } else {
        sj = 1;
    }

#elif (AMREX_SPACEDIM == 3)
    // For x-dimension
    if ((i - _i * ratio[0]) == 0) {
        si = -1;
    } else {
        si = 1;
    }
    // For y-dimension
    if ((j - _j * ratio[1]) == 0) {
        sj = -1;
    } else {
        sj = 1;
    }
    // For z-dimension
    if ((k - _k * ratio[2]) == 0) {
        sk = -1;
    } else {
        sk = 1;
    }

#endif

    // --- Retrieve Fine Sub-Cell Index (idx) ---
    int idx = -1; // Initialize to -1 to detect if a match is found
    for(int _idx_loop = 0; _idx_loop < std::pow(2,AMREX_SPACEDIM); ++_idx_loop)
    {
#if (AMREX_SPACEDIM == 1)
        if((si == amr_projmat_int[_idx_loop][0]))
#elif (AMREX_SPACEDIM == 2)
        if((si == amr_projmat_int[_idx_loop][0]) && (sj == amr_projmat_int[_idx_loop][1]))
#elif (AMREX_SPACEDIM == 3)
        if((si == amr_projmat_int[_idx_loop][0]) && (sj == amr_projmat_int[_idx_loop][1])
           && (sk == amr_projmat_int[_idx_loop][2]))
#endif
        {
            idx = _idx_loop;
            break;
        }
    }

    // Populate fine sub-cell index in _map
    _map.fidx = idx;

    return _map;
}

amrex::Vector<AmrDG::L2ProjInterp::IndexMap> AmrDG::L2ProjInterp::set_coarse_fine_idx_map(int i, int j, int k, const amrex::IntVect& ratio)
{
  //pass coarse cell index and return all fine cells indices and their
  //respective rf-element indices to lcoate them w.r.t coarse cell

  int num_overlap_cells = (int)std::pow(2,AMREX_SPACEDIM);

  amrex::Vector<IndexMap> _map;
  _map.resize(num_overlap_cells);

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
    for(int _idx=0; _idx<num_overlap_cells; ++_idx)
    { 
      if((si ==amr_projmat_int[_idx][0]))
      {
        idx=_idx;
        break;
      }
    }

    _map[idx].i = i_f;
    _map[idx].j = j_f;
    _map[idx].k = k_f;
    _map[idx].fidx = idx;

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
      for(int _idx=0; _idx<num_overlap_cells; ++_idx)
      { 
        if((si ==amr_projmat_int[_idx][0]) && (sj ==amr_projmat_int[_idx][1]))
        {
          idx=_idx;
          break;
        }
      }

      _map[idx].i = i_f;
      _map[idx].j = j_f;
      _map[idx].k = k_f;
      _map[idx].fidx = idx;

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
        for(int _idx=0; _idx<num_overlap_cells; ++_idx)
        { 
          if((si ==amr_projmat_int[_idx][0]) && (sj ==amr_projmat_int[_idx][1]) 
            && (sk ==amr_projmat_int[_idx][2]))
          {
            idx=_idx;
            break;
          }
        }

        _map[idx].i = i_f;
        _map[idx].j = j_f;
        _map[idx].k = k_f;
        _map[idx].fidx = idx;

      }
    }
  }
#endif  

  return _map;
}

