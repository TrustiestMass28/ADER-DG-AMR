#include "AmrDG.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>


void AmrDG::set_vandermat()
{
  for(int i=0; i<quadrule->qMp_st; ++i){
    for(int j=0; j<basefunc->Np_st; ++j){
      V[i][j] = basefunc->phi_st(j,basefunc->basis_idx_st,quadrule->xi_ref_quad_st[i]);//modPhi_j(x_i)==V_{ij}
      //V[i][j] =  modPhi(j, xi_ref_equidistant[i]);
    }
  } 

  //inverse vandermonde matrix
  Eigen::MatrixXd V_eigen(quadrule->qMp_st, basefunc->Np_st);
  for (int i = 0; i < quadrule->qMp_st; ++i) {
    for (int j = 0; j < basefunc->Np_st; ++j) {
        V_eigen(i, j) = V[i][j];
    }
  }
  
  //Eigen::MatrixXd Vinv_eigen(mNp,qMp);
  //Vinv_eigen= V_eigen.completeOrthogonalDecomposition().pseudoInverse(); 
  
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(V_eigen, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singularValues = svd.singularValues();  
  Eigen::MatrixXd Vinv_eigen = svd.matrixV() * singularValues.asDiagonal().inverse() 
                              * svd.matrixU().transpose();
 
  //Eigen::MatrixXd Vinv_eigen(mNp,qMp);
  //Vinv_eigen=((V_eigen.transpose()*V_eigen).inverse())*(V_eigen.transpose());
  
  for (int i = 0; i < basefunc->Np_st; ++i) {
    for (int j = 0; j < quadrule->qMp_st; ++j) {
        Vinv[i][j] = Vinv_eigen(i, j);
    }
  }
}

void AmrDG::set_ref_element_matrix()
{
  //Generate matrices used for predictor step
  for(int j=0; j<basefunc->Np_st;++j){
    
    for(int i=0; i<basefunc->Np_st;++i){
      Mk_h_w[j][i]= refMat_phiphi(i,j,true,false)*((basefunc->phi_t(j,1.0)*basefunc->phi_t(i,1.0))
                    -refMat_tphiDtphi(j,i));  
                    
      for(int d=0; d<AMREX_SPACEDIM; ++d){
        Sk_pred[d][j][i]   = refMat_tphitphi(i,j)*refMat_phiDphi(i,j,d);  
      }      
      Mk_pred_src[j][i] =refMat_tphitphi(i,j)*refMat_phiphi(i,j,true,false);
    }
    
    for(int i=0; i<basefunc->Np_s;++i){
      Mk_pred[j][i] = basefunc->phi_t(j,-1.0)*refMat_phiphi(i,j,true,true);
    }
  }

  Eigen::MatrixXd Sk_pred_eigen(basefunc->Np_st,basefunc->Np_st);
  Eigen::MatrixXd Mk_s_eigen(basefunc->Np_st,basefunc->Np_st);
  Eigen::MatrixXd Vinv_eigen(basefunc->Np_st,quadrule->qMp_st);
  Eigen::MatrixXd Sk_predVinv_eigen(basefunc->Np_st,quadrule->qMp_st);
  Eigen::MatrixXd Mk_sVinv_eigen(basefunc->Np_st,quadrule->qMp_st);
  
  for (int i = 0; i < basefunc->Np_st; ++i) {
    for (int j = 0; j < quadrule->qMp_st; ++j){
      Vinv_eigen(i, j) = Vinv[i][j];
    }
  }

  for(int i=0; i<basefunc->Np_st;++i){
    for(int j=0; j<basefunc->Np_st;++j){
      Mk_s_eigen(i,j)=Mk_pred_src[i][j]; 
    }
  }
  
  Mk_sVinv_eigen = Mk_s_eigen*Vinv_eigen;
  
  for (int i = 0; i < basefunc->Np_st; ++i) {
    for (int j = 0; j < quadrule->qMp_st; ++j) {
      Mk_pred_srcVinv[i][j] = Mk_sVinv_eigen(i, j);  
    }
  }

  for(int d=0; d<AMREX_SPACEDIM; ++d){
    for(int i=0; i<basefunc->Np_st;++i){
      for(int j=0; j<basefunc->Np_st;++j){
        Sk_pred_eigen(i,j)=Sk_pred[d][i][j];        
      }
    }
    
    Sk_predVinv_eigen = Sk_pred_eigen*Vinv_eigen;
    
    for (int i = 0; i < basefunc->Np_st; ++i) {
      for (int j = 0; j < quadrule->qMp_st; ++j) {
        Sk_predVinv[d][i][j] = Sk_predVinv_eigen(i, j);  
      }
    }    
  }

  Eigen::MatrixXd Mk_h_w_eigen(basefunc->Np_st, basefunc->Np_st);
  for (int i = 0; i < basefunc->Np_st; ++i) {
    for (int j = 0; j < basefunc->Np_st; ++j) {
      Mk_h_w_eigen(i, j) = Mk_h_w[i][j];
    }
  }
    
  Eigen::MatrixXd Mk_h_w_inv_eigen(basefunc->Np_st, basefunc->Np_st);
  Mk_h_w_inv_eigen = Mk_h_w_eigen.inverse(); 
 
  for (int i = 0; i < basefunc->Np_st; ++i) {
    for (int j = 0; j < basefunc->Np_st; ++j) {
      Mk_h_w_inv[i][j] = Mk_h_w_inv_eigen(i, j);
    }
  }
  
  //Generate matrices used for Gaussian quadrature, they contain in ADER-DG step  
  //using same i,j idx convention as in documentation for readability

  int N = quadrule->qMp_1d;  

  amrex::Real w;
  amrex::Real wm;
  amrex::Real wp;

  for(int j=0; j<basefunc->Np_s;++j){
    for(int i=0; i<basefunc->Np_s;++i){
      Mk_corr[j][i] = refMat_phiphi(i,j,false,false);
    }
  }  
  
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    for(int i=0; i<quadrule->qMp_st;++i){  
      w = 1.0;
      for(int d_=0; d_<AMREX_SPACEDIM+1; ++d_){
        w*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_st[i][d_]),2.0);
      }
      for(int j=0; j<basefunc->Np_s;++j){
        Sk_corr[d][j][i] = basefunc->dphi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_st[i],d)*w;  
        
        //TODO: correct to use xi_ref_quad_st? shoudln be ebtter to use xi_ref_quad_s 
        //      sicne loop over Np_s? is it also correct dphi_s or should it be dphi_st? BUG?
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
      for(int j=0; j<basefunc->Np_s;++j){
        Mkbd[2*d][j][i]   = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_st_bdm[d][i])*wm;
        Mkbd[2*d+1][j][i] = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_st_bdp[d][i])*wp;
      } 
    }
  }
   
  //TODO: BUG? Also here, is it correct the type of quad points used? why _st and not _s?
  for(int i=0; i<quadrule->qMp_st;++i){  
    w = 1.0;
    for(int d_=0; d_<AMREX_SPACEDIM+1; ++d_){
      w*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_st[i][d_]),2.0);
    }
    for(int j=0; j<basefunc->Np_s;++j){
      Mk_corr_src[j][i] = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_st[i])*w;
    }
  }
  
  //general volume integral quadrature matrix with only spatial nodes 
  //(i.e for only spatial integrals)
  //used for the BC,IC
  for(int i=0; i<quadrule->qMp_s;++i){  
    w = 1.0;
    for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
      w*=2.0/std::pow(std::assoc_legendre(N,1,quadrule->xi_ref_quad_s[i][d_]),2.0);
    }
    for(int j=0; j<basefunc->Np_s;++j){
      quadmat[j][i] = basefunc->phi_s(j,basefunc->basis_idx_s,quadrule->xi_ref_quad_s[i])*w;
    }
  }  

}

amrex::Real AmrDG::refMat_phiphi(int i,int j, bool is_predictor, bool is_mixed_nmodes) const 
{
  //computes M_{ji}=\int_{[-1,1]^D} \phi_i*\phi_j dx
  amrex::Real m= 1.0;
  if(is_predictor)
  {  
    if(is_mixed_nmodes)
    {
      //compute mass matrix for integral of spatial only basis functions 
      //(which are a tensor product of 1d Legendre polynomials)
      //utilizes indexing of modified basis function and of classic basis function, 
      //used in predictor U_w term
      for(int d=0; d<AMREX_SPACEDIM; ++d){
        m*=(amrex::Real)kroneckerDelta(basefunc->basis_idx_s[i][d],basefunc->basis_idx_st[j][d])
            *(2.0/(2.0*(amrex::Real)basefunc->basis_idx_st[j][d]+1.0));
      }     
    }
    else
    {
      //compute mass matrix for integral of spatial only basis functions 
      //(which are a tensor product of 1d Legendre polynomials)
      //utilizes indexing of modified basis function
      for(int d=0; d<AMREX_SPACEDIM; ++d){
        m*=(amrex::Real)kroneckerDelta(basefunc->basis_idx_st[i][d],basefunc->basis_idx_st[j][d])
            *(2.0/(2.0*(amrex::Real)basefunc->basis_idx_st[j][d]+1.0));
      } 
    }
  }
  else
  {
    //compute mass matrix for integral of spatial only basis functions
    //(which are a tensor product of 1d Legendre polynomials)
    //utilizes indexing of classic basis fucntion 
    for(int d=0; d<AMREX_SPACEDIM; ++d){
      m*=(amrex::Real)kroneckerDelta(basefunc->basis_idx_s[i][d],basefunc->basis_idx_s[j][d])
          *(2.0/(2.0*(amrex::Real)basefunc->basis_idx_s[j][d]+1.0));
    }
  }

  return m;
}

amrex::Real AmrDG::refMat_phiDphi(int i,int j, int dim) const 
{
  //computes Sd_{ji}=\int_{[-1,1]^D} \phi_i*d/dx_d \phi_j dx
  
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
  
  //
  //computes the integral using gaussian quadrature
  int N = quadrule->qMp_1d;
  amrex::Real w;
  amrex::Real sum=0.0;
  for(int q=0; q<(int)std::pow(N,AMREX_SPACEDIM);++q){  
    w = 1.0;
    amrex::Real phi = 1.0; 
    for  (int d = 0; d < AMREX_SPACEDIM; ++d){
      phi*=std::legendre(basefunc->basis_idx_st[j][d], quadrule->xi_ref_quad_s[q][d]);
    }
    
    amrex::Real dphi = 1.0;
    for  (int a = 0; a < AMREX_SPACEDIM; ++a){
      if(a!=dim)
      {
        dphi*=std::legendre(basefunc->basis_idx_st[i][a], quadrule->xi_ref_quad_s[q][a]);
      }
      else
      {
        dphi*=(std::assoc_legendre(basefunc->basis_idx_st[i][dim],1,quadrule->xi_ref_quad_s[q][dim]))
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

amrex::Real AmrDG::refMat_tphitphi(int i,int j) const 
{
  //computes t_M_{ji}=\int_{[-1,1]^D} P_i*P_j dx
  //compute mass matrix for integral of temporal only basis functions
  //currently we use also for time Legendre polynomials, but in theory 
  //the reference integral of any
  //basis function can be implemented here
  //index[-1] indicates time coordinate
  
  return (amrex::Real)kroneckerDelta(basefunc->basis_idx_t[i][AMREX_SPACEDIM],
    basefunc->basis_idx_t[j][AMREX_SPACEDIM])
    *(2.0/(2.0*(amrex::Real)basefunc->basis_idx_t[j][AMREX_SPACEDIM]+1.0));
}

amrex::Real AmrDG::refMat_tphiDtphi(int i,int j) const 
{
  //computes Sd_{ji}=\int_{[-1,1]^D} P_i(t)*d/dt P_j(t) dt
  
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
    tphiDtphi+=(basefunc->phi_t(j, quadrule->xi_ref_quad_t[q][0])*basefunc->dtphi_t(i, quadrule->xi_ref_quad_t[q][0])*w);  
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

/*

Real AmrDG::RefMat_phiDphi(int i,int j, int dim) const
{ 

  //
}


*/