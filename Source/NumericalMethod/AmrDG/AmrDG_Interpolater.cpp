#include "AmrDG.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

using namespace amrex;

void AmrDG::L2ProjInterp::reflux(amrex::MultiFab& U_crse,
                               const amrex::MultiFab& correction_mf,
                               int lev,
                               const amrex::Geometry& crse_geom) noexcept
{
    auto _mesh =  numme->mesh.lock();

    amrex::Real vol = _mesh->get_dvol(lev);
    amrex::Real inv_jac = std::pow(2.0, AMREX_SPACEDIM) / vol;

    // Time: [-1, 1] -> [0, Dt]   => Factor: Dt / 2.0
    // Space: [-1, 1]^(D-1) -> Face Area => Factor: dvol / 2^(D-1)
    // amrex::Real jacobian = (Dt / 2.0) * (dvol / std::pow(2.0, AMREX_SPACEDIM - 1));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(correction_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            amrex::Array4<const amrex::Real> corr = correction_mf.const_array(mfi);
            amrex::Array4<amrex::Real> u_crse = U_crse.array(mfi);

            // Reuse precomputed C-F interface face data (face-centered, per dimension)
            amrex::Vector<amrex::Array4<const int>> bc_arr(AMREX_SPACEDIM);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                bc_arr[d] = numme->cf_face_b_coarse(lev,d).const_array(mfi);
            }

            amrex::ParallelFor(bx, [=] (int i, int j, int k) noexcept
            {
              // A cell is at the C-F interface if any of its surrounding faces
              // (lo or hi in any dimension) was flagged in cf_face_b_coarse
              bool is_at_interface = false;
              int shift[] = {0,0,0};
              for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                  shift[d] = 1;
                  // lo face = (i,j,k), hi face = (i+shift[0], j+shift[1], k+shift[2])
                  if (bc_arr[d](i,j,k) != 0 ||
                      bc_arr[d](i+shift[0], j+shift[1], k+shift[2]) != 0) {
                      is_at_interface = true;
                      break;
                  }
                  shift[d] = 0;
              }

              if (is_at_interface)
              {
                  Eigen::VectorXd f_delta(numme->Np_s);
                  for(int n=0; n<numme->Np_s; ++n){
                      f_delta(n) = corr(i,j,k,n);
                  }

                  Eigen::VectorXd delta_u_w = inv_jac * (Minv * f_delta);

                  for (int n = 0; n < numme->Np_s; ++n) {
                      u_crse(i, j, k, n) += delta_u_w(n);
                  }
              }
            });
        }
    }
}

const Eigen::MatrixXd& AmrDG::L2ProjInterp::get_flux_proj_mat(int d, int child_idx, int b) const {
    if (b == 1) {
        // Coarse cell is to the left, we need the High-side projection
        return P_flux_fc_high[d][child_idx];
    } else {
        // Default or Coarse cell to the right, use Low-side projection
        return P_flux_fc_low[d][child_idx];
    }
}

void AmrDG::L2ProjInterp::flux_proj_mat()
{
    #define FLUX_PROJ_CASE(PP) case PP: _flux_proj_mat<PP>(); break;
    switch(numme->p) {
        FLUX_PROJ_CASE(1) FLUX_PROJ_CASE(2) FLUX_PROJ_CASE(3) FLUX_PROJ_CASE(4) FLUX_PROJ_CASE(5)
        FLUX_PROJ_CASE(6) FLUX_PROJ_CASE(7) FLUX_PROJ_CASE(8) FLUX_PROJ_CASE(9) FLUX_PROJ_CASE(10)
        default: amrex::Abort("Unsupported polynomial order p");
    }
    #undef FLUX_PROJ_CASE
}

template<int P>
void AmrDG::L2ProjInterp::_flux_proj_mat()
{
    // Setup dimensions
    int num_face_children = (int)std::pow(2, AMREX_SPACEDIM - 1);
    int num_q_pts = numme->quadrule->qMp_st_bd;

    // Resize Outer Vectors [Direction]
    P_flux_fc_low.resize(AMREX_SPACEDIM);
    P_flux_fc_high.resize(AMREX_SPACEDIM);

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {

        // Resize Inner Vectors [Child Index]
        P_flux_fc_low[d].resize(num_face_children);
        P_flux_fc_high[d].resize(num_face_children);

        for (int k = 0; k < num_face_children; ++k) {

            // Resize the Eigen Matrices (Rows: Coarse Modes, Cols: Fine Quad Pts)
            P_flux_fc_low[d][k].resize(AmrDG::BasisLegendre<P>::Np_s, num_q_pts);
            P_flux_fc_low[d][k].setZero();

            P_flux_fc_high[d][k].resize(AmrDG::BasisLegendre<P>::Np_s, num_q_pts);
            P_flux_fc_high[d][k].setZero();

            // Determine Tangential Shifts for this child 'k'
            amrex::Vector<amrex::Real> t_shifts(AMREX_SPACEDIM, 0.0);
            int temp_k = k;
            int bit_counter = 0;
            for(int dir=0; dir<AMREX_SPACEDIM; ++dir) {
                if (dir == d) continue;
                int is_upper = (temp_k >> bit_counter) & 1;
                t_shifts[dir] = (is_upper) ? 0.5 : -0.5;
                bit_counter++;
            }

            // Fill Matrix Elements using QuadratureGaussLegendre lookups
            for (int r = 0; r < AmrDG::BasisLegendre<P>::Np_s; ++r) {
                const auto& mi = AmrDG::MultiIndex<P, AMREX_SPACEDIM>::table[r];
                for (int m = 0; m < num_q_pts; ++m) {
                    // phi_s at coarse boundary: normal dim d has xi=±1,
                    // tangential dims use shifted GL nodes (0.5*node + shift)
                    double phi_low = 1.0;
                    double phi_high = 1.0;
                    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
                        if (dim == d) {
                            phi_low  *= AmrDG::QuadratureGaussLegendre<P>::bd_val[mi.idx[dim]][0]; // xi=-1
                            phi_high *= AmrDG::QuadratureGaussLegendre<P>::bd_val[mi.idx[dim]][1]; // xi=+1
                        } else {
                            // Tangential dim: 0.5*nodes[q_a] + t_shifts[dim]
                            int pos = AmrDG::QuadratureGaussLegendre<P>::bd_free_pos(dim, d);
                            int q_a = AmrDG::QuadratureGaussLegendre<P>::node_idx(m, pos, AMREX_SPACEDIM);
                            const auto& tbl = (t_shifts[dim] < 0.0)
                                ? AmrDG::QuadratureGaussLegendre<P>::shifted_lo_val
                                : AmrDG::QuadratureGaussLegendre<P>::shifted_hi_val;
                            phi_low  *= tbl[mi.idx[dim]][q_a];
                            phi_high *= tbl[mi.idx[dim]][q_a];
                        }
                    }

                    amrex::Real w_m = numme->quad_weights_st_bdm[d][m];
                    amrex::Real w_p = numme->quad_weights_st_bdp[d][m];

                    P_flux_fc_low[d][k](r, m)  = phi_low  * w_m;
                    P_flux_fc_high[d][k](r, m) = phi_high * w_p;
                }
            }
        }
    }
}

void AmrDG::L2ProjInterp::interp_proj_mat()
{
    #define INTERP_PROJ_CASE(PP) case PP: _interp_proj_mat<PP>(); break;
    switch(numme->p) {
        INTERP_PROJ_CASE(1) INTERP_PROJ_CASE(2) INTERP_PROJ_CASE(3) INTERP_PROJ_CASE(4) INTERP_PROJ_CASE(5)
        INTERP_PROJ_CASE(6) INTERP_PROJ_CASE(7) INTERP_PROJ_CASE(8) INTERP_PROJ_CASE(9) INTERP_PROJ_CASE(10)
        default: amrex::Abort("Unsupported polynomial order p");
    }
    #undef INTERP_PROJ_CASE
}

template<int P>
void AmrDG::L2ProjInterp::_interp_proj_mat()
{
  constexpr int num_overlap_cells = 1 << AMREX_SPACEDIM;

  //coarse->fine projection matrix
  P_cf.resize(num_overlap_cells);
  for (int i = 0; i < num_overlap_cells; ++i) {
      P_cf[i].resize(AmrDG::BasisLegendre<P>::Np_s, AmrDG::BasisLegendre<P>::Np_s);
      P_cf[i].setZero();
  }

  //fine->coarse projection matrix
  P_fc.resize(num_overlap_cells);
  for (int i = 0; i < num_overlap_cells; ++i) {
      P_fc[i].resize(AmrDG::BasisLegendre<P>::Np_s, AmrDG::BasisLegendre<P>::Np_s);
      P_fc[i].setZero();
  }

  //mass matrix
  M.resize(AmrDG::BasisLegendre<P>::Np_s, AmrDG::BasisLegendre<P>::Np_s);
  M.setZero();

  //Compute mass matrices
  for(int j=0; j<AmrDG::BasisLegendre<P>::Np_s;++j){
    for(int n=0; n<AmrDG::BasisLegendre<P>::Np_s;++n){
      M(j,n)= numme->refMat_phiphi(j,numme->basis_idx_s,n,numme->basis_idx_s);
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singularValues = svd.singularValues();
  Minv = svd.matrixV() * singularValues.asDiagonal().inverse()
                              * svd.matrixU().transpose();

  //Compute projection matrices for each sub-cell (indicated by idx)
  for(int l=0; l<std::pow(2,AMREX_SPACEDIM); ++l)
  {
    amrex::Real shift[AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int bit = (l >> (AMREX_SPACEDIM - 1 - d)) & 1;
        shift[d] = bit ? 0.5 : -0.5;
    }

    for(int j=0; j<AmrDG::BasisLegendre<P>::Np_s;++j){
      for(int m=0; m<AmrDG::BasisLegendre<P>::Np_s;++m){

        amrex::Real sum_cf = 0.0;
        amrex::Real sum_fc = 0.0;
        const auto& mi_m = AmrDG::MultiIndex<P, AMREX_SPACEDIM>::table[m];
        const auto& mi_j = AmrDG::MultiIndex<P, AMREX_SPACEDIM>::table[j];
        for(int q=0; q<numme->quadrule->qMp_s; ++q)
        {
          // phi_s(idx, 0.5*nodes[q_d] + shift[d]) = prod_d P_{mi[d]}(0.5*nodes[q_d] + shift[d])
          // shift[d] = -0.5 → shifted_lo_val, shift[d] = +0.5 → shifted_hi_val
          double phi_m = 1.0;
          double phi_j = 1.0;
          for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            int q_d = AmrDG::QuadratureGaussLegendre<P>::node_idx(q, d, AMREX_SPACEDIM);
            const auto& tbl = (shift[d] < 0.0) ? AmrDG::QuadratureGaussLegendre<P>::shifted_lo_val
                                                : AmrDG::QuadratureGaussLegendre<P>::shifted_hi_val;
            phi_m *= tbl[mi_m.idx[d]][q_d];
            phi_j *= tbl[mi_j.idx[d]][q_d];
          }

          sum_cf += numme->quadmat(j,q) * phi_m;
          sum_fc += numme->quadmat(m,q) * phi_j;
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
  sum.setZero();
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
    //and sub-cell index locating fine cell w.r.t coarse cell
    IndexMap _map;
    IntVect fine{AMREX_D_DECL(i, j, k)};
    IntVect crse = amrex::coarsen(fine, ratio);

    _map.i = crse[0];
    _map.j = (AMREX_SPACEDIM > 1) ? crse[1] : 0;
    _map.k = (AMREX_SPACEDIM > 2) ? crse[2] : 0;

    // Sub-cell index from bit pattern: bit_d = fine[d] - crse[d]*ratio[d] (0 or 1)
    // idx = sum_d bit_d * 2^(D-1-d), matching amr_projmat_int layout
    int idx = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int bit = fine[d] - crse[d] * ratio[d];
        idx += bit * (1 << (AMREX_SPACEDIM - 1 - d));
    }
    _map.fidx = idx;

    return _map;
}

amrex::Vector<AmrDG::L2ProjInterp::IndexMap> AmrDG::L2ProjInterp::set_coarse_fine_idx_map(int i, int j, int k, const amrex::IntVect& ratio)
{
  //pass coarse cell index and return all fine cells indices and their
  //respective sub-cell indices to locate them w.r.t coarse cell

  constexpr int num_children = 1 << AMREX_SPACEDIM;
  amrex::Vector<IndexMap> _map(num_children);

  IntVect base{AMREX_D_DECL(i * ratio[0], j * ratio[1], k * ratio[2])};

  for (int idx = 0; idx < num_children; ++idx) {
      IntVect fine = base;
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          fine[d] += (idx >> (AMREX_SPACEDIM - 1 - d)) & 1;
      }
      _map[idx].i = fine[0];
      _map[idx].j = (AMREX_SPACEDIM > 1) ? fine[1] : 0;
      _map[idx].k = (AMREX_SPACEDIM > 2) ? fine[2] : 0;
      _map[idx].fidx = idx;
  }

  return _map;
}

