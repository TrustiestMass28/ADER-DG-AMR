#include "AmrDG.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>


void AmrDG::set_vandermat()
{
  for(int i=0; i<quadrule->qMp_st; ++i){
    for(int j=0; j<basefunc->Np_st; ++j){
      V(i,j) = basefunc->phi_st(j,basefunc->basis_idx_st,quadrule->xi_ref_quad_st[i]);//modPhi_j(x_i)==V_{ij}
    }
  }

  //inverse vandermonde matrix via SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singularValues = svd.singularValues();
  Vinv = svd.matrixV() * singularValues.asDiagonal().inverse()
       * svd.matrixU().transpose();
}

void AmrDG::set_ref_element_matrix()
{
  //Generate matrices used for predictor step
  for(int j=0; j<basefunc->Np_st;++j){

    for(int i=0; i<basefunc->Np_st;++i){
      Mk_h_w(j,i)= refMat_phiphi(j,basefunc->basis_idx_st,i,basefunc->basis_idx_st)*((basefunc->phi_t(j,1.0)*basefunc->phi_t(i,1.0))
                    -refMat_tphiDtphi(j,i));

      for(int d=0; d<AMREX_SPACEDIM; ++d){
        Sk_pred[d](j,i)   = refMat_tphitphi(j,i)*refMat_phiDphi(j,basefunc->basis_idx_st,i,basefunc->basis_idx_st,d);
      }
      Mk_pred_src(j,i) =refMat_tphitphi(j,i)*refMat_phiphi(j,basefunc->basis_idx_st,i,basefunc->basis_idx_st);
    }

    for(int i=0; i<basefunc->Np_s;++i){
      Mk_pred(j,i) = basefunc->phi_t(j,-1.0)*refMat_phiphi(j,basefunc->basis_idx_st,i,basefunc->basis_idx_s);
    }
  }

  //Compute pre-multiplied matrices: Mk_pred_srcVinv = Mk_pred_src * Vinv
  Mk_pred_srcVinv = Mk_pred_src * Vinv;

  //Compute pre-multiplied matrices: Sk_predVinv[d] = Sk_pred[d] * Vinv
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    Sk_predVinv[d] = Sk_pred[d] * Vinv;
  }

  //Compute inverse of Mk_h_w
  Mk_h_w_inv = Mk_h_w.inverse();
  
  //Generate matrices used for Gaussian quadrature, they contain in ADER-DG step  
  //using same i,j idx convention as in documentation for readability

  int N = quadrule->qMp_1d;  

  amrex::Real w;
  amrex::Real wm;
  amrex::Real wp;

  for(int j=0; j<basefunc->Np_s;++j){
    for(int i=0; i<basefunc->Np_s;++i){
      Mk_corr(j,i) = refMat_phiphi(j,basefunc->basis_idx_s,i,basefunc->basis_idx_s);
    }
  }

  //Here, use qMp_st because ists quadrature of a double int_t int_dx integral
  //therefore has D+1 pts
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    for(int i=0; i<quadrule->qMp_st;++i){
      w = 1.0;
      for(int d_=0; d_<AMREX_SPACEDIM+1; ++d_){
        w*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_st[i][d_]),2.0);
      }
      for(int j=0; j<basefunc->Np_s;++j){
        Sk_corr[d](j,i) = basefunc->dphi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_st[i],d)*w;
      }
    }
    for(int i=0; i<quadrule->qMp_st_bd;++i){
      wm = 1.0;
      wp = 1.0;
      for(int d_=0; d_<AMREX_SPACEDIM+1; ++d_){
        if(d_!=d)
        {
          wm*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_st_bdm[d][i][d_]),2.0);
          wp*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_st_bdp[d][i][d_]),2.0);
        }
      }

      quad_weights_st_bdm[d][i] = wm;
      quad_weights_st_bdp[d][i] = wp;

      for(int j=0; j<basefunc->Np_s;++j){
        Mkbdm[d](j,i) = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_st_bdm[d][i])*wm;
        Mkbdp[d](j,i) = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_st_bdp[d][i])*wp;
      }
    }
  }

  for(int i=0; i<quadrule->qMp_st;++i){
    w = 1.0;
    for(int d=0; d<AMREX_SPACEDIM+1; ++d){
      w*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_st[i][d]),2.0);
    }
    for(int j=0; j<basefunc->Np_s;++j){
      Mk_corr_src(j,i) = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_st[i])*w;
    }
  }

  //general volume integral quadrature matrix with only spatial nodes
  //(i.e for only spatial integrals)
  //used for the BC,IC
  for(int i=0; i<quadrule->qMp_s;++i){
    w = 1.0;
    for(int d=0; d<AMREX_SPACEDIM; ++d){
      w*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_s[i][d]),2.0);
    }
    for(int j=0; j<basefunc->Np_s;++j){
      quadmat(j,i) = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_s[i])*w;
    }
  }

  //general surface integral quadrature matrix with only spatial boundary nodes
  //used for flux registers
  for(int i=0; i<quadrule->qMp_s_bd;++i){
    for(int d=0; d<AMREX_SPACEDIM; ++d){
      wm = 1.0;
      for(int d_=0; d_<AMREX_SPACEDIM+1; ++d_){
        if(d_!=d)
        {
          wm*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_s_bdm[d][i][d_]),2.0);
        }

        for(int j=0; j<basefunc->Np_s;++j){
          quadmat_bd[d](j,i) = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_s_bdm[d][i])*wm;
        }
      }
    }
  }
}

amrex::Real AmrDG::refMat_phiphi(int j, const amrex::Vector<amrex::Vector<int>>& idx_map_j, 
                                 int i, const amrex::Vector<amrex::Vector<int>>& idx_map_i) const 
{
  //computes M_{ji}=M_{ij}=\int_{[-1,1]^D} \phi_i*\phi_j dx

  amrex::Real m= 1.0;
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    m*=(amrex::Real)kroneckerDelta(idx_map_i[i][d],idx_map_j[j][d])
        *(2.0/(2.0*(amrex::Real)idx_map_j[j][d]+1.0));
  }    

  return m;
}

amrex::Real AmrDG::refMat_phiDphi(int j, const amrex::Vector<amrex::Vector<int>>& idx_map_j,
                                  int i, const amrex::Vector<amrex::Vector<int>>& idx_map_i,
                                  int dim) const 
{
  //computes Sd_{ji}=\int_{[-1,1]^D} \phi_j*d/dx_d \phi_i dx
  
  //computes the integral using analytical form
  //amrex::Real m1= 1.0;
  //amrex::Real m2= 0.0;
  //amrex::Real m3= 0.0;

  //for(int d=0; d<AMREX_SPACEDIM; ++d){
  //  if(d != dim)
  //  {
  //    m1*=(amrex::Real)KroneckerDelta(mat_idx_st[j][d],mat_idx_st[i][d])
  //      *(2.0/(2.0*(amrex::Real)mat_idx_st[i][d]+1.0));
  //  }
  //}
  
  //int l = mat_idx_st[i][dim]+1;
  //for(int k=0; k<=l; ++k){
  //  m2+=Coefficient_c(k,l)*(amrex::Real)KroneckerDelta(mat_idx_st[j][dim],k)
  //    *(2.0/(2.0*(amrex::Real)k+1.0));
  //}
  //m2*=0.5;

  //m3 = 0.5*(amrex::Real)l*((amrex::Real)l-1.0)*(2.0/(2.0*(amrex::Real)l+1.0))
  //    *(amrex::Real)KroneckerDelta(mat_idx_st[j][dim],l);
  
  //return m1*(m2+m3);
  
  //computes the integral using gaussian quadrature
  int N = quadrule->qMp_1d;
  amrex::Real w;
  amrex::Real sum=0.0;
  for(int q=0; q<(int)std::pow(N,AMREX_SPACEDIM);++q){  
    //since is a spatial integral, use purely spatial quadrature points: quadrule->xi_ref_quad_s
    w = 1.0;
    amrex::Real phi = 1.0; 
    for  (int d = 0; d < AMREX_SPACEDIM; ++d){
      phi*=std::legendre(idx_map_j[j][d], quadrule->xi_ref_quad_s[q][d]);
    }
    
    amrex::Real dphi = 1.0;
    for  (int a = 0; a < AMREX_SPACEDIM; ++a){
      if(a!=dim)
      {
        dphi*=std::legendre(idx_map_i[i][a], quadrule->xi_ref_quad_s[q][a]);
      }
      else
      {
        dphi*=(std::assoc_legendre(idx_map_i[i][dim],1,quadrule->xi_ref_quad_s[q][dim]))
            /(std::sqrt(1.0-std::pow(quadrule->xi_ref_quad_s[q][dim],2.0)));
      }   
    }
    
    for(int d=0; d<AMREX_SPACEDIM; ++d){
      w*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_s[q][d]),2.0);
    }
    sum+=(phi*dphi*w);   
  }

  return sum; 
}

amrex::Real AmrDG::refMat_tphitphi(int j,int i) const 
{
  //computes t_M_{ji}=\int_{[-1,1]^D} P_i*P_j dx
  //compute mass matrix for integral of temporal only basis functions
  //currently we use also for time Legendre polynomials, but in theory 
  //the reference integral of any
  //basis function can be implemented here
  //index[-1] indicates time coordinate
  
  //NB:basis_idx_st[ctr][AMREX_SPACEDIM] == basis_idx_t[ctr][0];  

  return (amrex::Real)kroneckerDelta(basefunc->basis_idx_t[i][0],
          basefunc->basis_idx_t[j][0])
          *(2.0/(2.0*(amrex::Real)basefunc->basis_idx_t[j][0]+1.0));
}

amrex::Real AmrDG::refMat_tphiDtphi(int j,int i) const 
{
  //computes Sd_{ji}=\int_{[-1,1]^D} P_i(t)*d/dt P_j(t) dt
  //component of Mh_ji
  
  ////computes the integral using analytical form
  //amrex::Real m2= 0.0;
  //amrex::Real m3= 0.0;
  
  //int l = mat_idx_st[i][AMREX_SPACEDIM]+1;
  //for(int k=0; k<=l; ++k){
  //  m2+=Coefficient_c(k,l)*(amrex::Real)KroneckerDelta(mat_idx_st[j][AMREX_SPACEDIM],k)
  //      *(2.0/(2.0*(amrex::Real)k+1.0));
  //}
  //m2*=0.5;

  //m3 = 0.5*(amrex::Real)l*((amrex::Real)l-1.0)*(2.0/(2.0*(amrex::Real)l+1.0))
  //    *(amrex::Real)KroneckerDelta(mat_idx_st[j][AMREX_SPACEDIM],l);
  
  //return (m2+m3);
  
 
  //computes the integral using gaussian quadrature
  int N = quadrule->qMp_1d; 
  amrex::Real w;
  amrex::Real tphiDtphi=0.0;
  for(int q=0; q<N;++q){  
    w = 1.0;
    w*=2.0/(amrex::Real)std::pow((amrex::Real)std::assoc_legendre(N,1,quadrule->xi_ref_quad_t[q][0]),2.0);
    tphiDtphi+=(basefunc->phi_t(i, quadrule->xi_ref_quad_t[q][0])*basefunc->dtphi_t(j, quadrule->xi_ref_quad_t[q][0])*w);  
  }
  return tphiDtphi;
  
}

int AmrDG::kroneckerDelta(int a, int b) const
{
  int k;
  if(a==b){k=1;}
  else{k=0;}
  return k;
}

Real AmrDG::coefficient_c(int k,int l) const
{ 
  if(k==l)
  {
    return -(amrex::Real)l*((amrex::Real)l-1.0);
  }
  else
  {
    return (2.0*(amrex::Real)k+1.0)*(1.0+std::pow(-1.0,k+l));
  }  
}
