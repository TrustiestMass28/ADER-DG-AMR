#include "AmrDG.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

void AmrDG::set_vandermat()
{}

void AmrDG::set_inv_vandermat()
{}

void AmrDG::set_ref_element_matrix()
{}


/*
void AmrDG::MatrixGenerator()
{
  //Generate matrices used for predictor step
  for(int j=0; j<mNp;++j){
    for(int i=0; i<mNp;++i){
      Mk_h_w[j][i]= RefMat_phiphi(i,j,true,false)*((tphi(j,1.0)*tphi(i,1.0))
                    -RefMat_tphiDtphi(j,i));   
                    
      for(int d=0; d<AMREX_SPACEDIM; ++d){
        Sk_pred[d][j][i]   = RefMat_tphitphi(i,j)*RefMat_phiDphi(i,j,d);  
      }      
      Mk_s[j][i] =RefMat_tphitphi(i,j)*RefMat_phiphi(i,j,true,false);
    }
        
    for(int i=0; i<Np;++i){
      Mk_pred[j][i] = tphi(j,-1.0)*RefMat_phiphi(i,j,true,true);
    }
    
  }

  Eigen::MatrixXd Sk_pred_eigen(mNp,mNp);
  Eigen::MatrixXd Mk_s_eigen(mNp,mNp);
  Eigen::MatrixXd Vinv_eigen(mNp,qMp);
  Eigen::MatrixXd Sk_predVinv_eigen(mNp,qMp);
  Eigen::MatrixXd Mk_sVinv_eigen(mNp,qMp);
  
  for (int i = 0; i < mNp; ++i) {
    for (int j = 0; j < qMp; ++j){
      Vinv_eigen(i, j) = Vinv[i][j];
    }
  }

  for(int i=0; i<mNp;++i){
    for(int j=0; j<mNp;++j){
      Mk_s_eigen(i,j)=Mk_s[i][j]; 
    }
  }
  
  Mk_sVinv_eigen = Mk_s_eigen*Vinv_eigen;
  
  for (int i = 0; i < mNp; ++i) {
    for (int j = 0; j < qMp; ++j) {
      Mk_sVinv[i][j] = Mk_sVinv_eigen(i, j);  
    }
  }

  for(int d=0; d<AMREX_SPACEDIM; ++d){
    for(int i=0; i<mNp;++i){
      for(int j=0; j<mNp;++j){
        Sk_pred_eigen(i,j)=Sk_pred[d][i][j];        
      }
    }
    
    Sk_predVinv_eigen = Sk_pred_eigen*Vinv_eigen;
    
    for (int i = 0; i < mNp; ++i) {
      for (int j = 0; j < qMp; ++j) {
        Sk_predVinv[d][i][j] = Sk_predVinv_eigen(i, j);  
      }
    }    
  }

  Eigen::MatrixXd Mk_h_w_eigen(mNp, mNp);
  for (int i = 0; i < mNp; ++i) {
    for (int j = 0; j < mNp; ++j) {
      Mk_h_w_eigen(i, j) = Mk_h_w[i][j];
    }
  }
    
  Eigen::MatrixXd Mk_h_w_inv_eigen(mNp, mNp);
  Mk_h_w_inv_eigen = Mk_h_w_eigen.inverse(); 
 
  for (int i = 0; i < mNp; ++i) {
    for (int j = 0; j < mNp; ++j) {
      Mk_h_w_inv[i][j] = Mk_h_w_inv_eigen(i, j);
    }
  }

  //Generate matrices used for Gaussian quadrature, they contain in ADER-DG step  
  //using same i,j idx convention as in documentation for readability
  int N = qMp_1d;
  amrex::Real w;
  amrex::Real wm;
  amrex::Real wp;
  for(int j=0; j<Np;++j){
    for(int i=0; i<Np;++i){
      Mk_corr[j][i] = RefMat_phiphi(i,j,false,false);
    }
  }  
  
  for(int d=0; d<AMREX_SPACEDIM; ++d){
    for(int i=0; i<qMp;++i){  
      w = 1.0;
      for(int d_=0; d_<AMREX_SPACEDIM+1; ++d_){
        w*=2.0/std::pow(std::assoc_legendre(N,1,xi_ref_GLquad[i][d_]),2.0);
      }
      for(int j=0; j<Np;++j){
        Sk_corr[d][j][i] = DPhi(j, xi_ref_GLquad[i], d)*w;
      }
    }
    for(int i=0; i<qMpbd;++i){
      wm = 1.0;
      wp = 1.0;
      for(int d_=0; d_<AMREX_SPACEDIM+1; ++d_){
        if(d_!=d)
        {
          wm*=2.0/std::pow(std::assoc_legendre(N,1,xi_ref_GLquad_bdm[d][i][d_]),2.0);
          wp*=2.0/std::pow(std::assoc_legendre(N,1,xi_ref_GLquad_bdp[d][i][d_]),2.0);   
        }
      }
      for(int j=0; j<Np;++j){
        Mkbd[2*d][j][i]   = Phi(j,xi_ref_GLquad_bdm[d][i])*wm;
        Mkbd[2*d+1][j][i] = Phi(j,xi_ref_GLquad_bdp[d][i])*wp;
      } 
    }
  }
   
  for(int i=0; i<qMp;++i){  
    w = 1.0;
    for(int d_=0; d_<AMREX_SPACEDIM+1; ++d_){
      w*=2.0/std::pow(std::assoc_legendre(N,1,xi_ref_GLquad[i][d_]),2.0);
    }
    for(int j=0; j<Np;++j){
      volquadmat[j][i] = Phi(j, xi_ref_GLquad[i])*w;
    }
  }
  
  //general volume integral quadrature matrix with only spatial nodes 
  //(i.e for only spatial integrals)
  //used for the BC,IC
  for(int i=0; i<qMp_L2proj;++i){  
    w = 1.0;
    for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
      w*=2.0/std::pow(std::assoc_legendre(N,1,xi_ref_GLquad_L2proj[i][d_]),2.0);
    }
    for(int j=0; j<Np;++j){
      L2proj_quadmat[j][i] = Phi(j, xi_ref_GLquad_L2proj[i])*w;
    }
  }
}

Real AmrDG::RefMat_phiphi(int i,int j, bool is_predictor, bool is_mixed_nmodes) const 
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
        m*=(amrex::Real)KroneckerDelta(mat_idx_s[i][d],mat_idx_st[j][d])
            *(2.0/(2.0*(amrex::Real)mat_idx_st[j][d]+1.0));
      }     
    }
    else
    {
      //compute mass matrix for integral of spatial only basis functions 
      //(which are a tensor product of 1d Legendre polynomials)
      //utilizes indexing of modified basis function
      for(int d=0; d<AMREX_SPACEDIM; ++d){
        m*=(amrex::Real)KroneckerDelta(mat_idx_st[i][d],mat_idx_st[j][d])
            *(2.0/(2.0*(amrex::Real)mat_idx_st[j][d]+1.0));
      } 
    }
  }
  else
  {
    //compute mass matrix for integral of spatial only basis functions
    //(which are a tensor product of 1d Legendre polynomials)
    //utilizes indexing of classic basis fucntion 
    for(int d=0; d<AMREX_SPACEDIM; ++d){
      m*=(amrex::Real)KroneckerDelta(mat_idx_s[i][d],mat_idx_s[j][d])
          *(2.0/(2.0*(amrex::Real)mat_idx_s[j][d]+1.0));
    }
  }

  return m;
}

Real AmrDG::RefMat_tphitphi(int i,int j) const
{
  //computes t_M_{ji}=\int_{[-1,1]^D} P_i*P_j dx
  //compute mass matrix for integral of temporal only basis functions
  //currently we use also for time Legendre polynomials, but in theory 
  //the reference integral of any
  //basis function can be implemented here
  //index[-1] indicates time coordinate
  
  return (amrex::Real)KroneckerDelta(mat_idx_st[i][AMREX_SPACEDIM],
          mat_idx_st[j][AMREX_SPACEDIM])
          *(2.0/(2.0*(amrex::Real)mat_idx_st[j][AMREX_SPACEDIM]+1.0));

}

Real AmrDG::RefMat_tphiDtphi(int i,int j) const
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
  int N = qMp_1d;
  amrex::Real w;
  amrex::Real tphiDtphi=0.0;
  for(int q=0; q<N;++q){  
    w = 1.0;
    w*=2.0/(amrex::Real)std::pow((amrex::Real)std::assoc_legendre(N,1,xi_ref_GLquad_t[q][0]),2.0);
    tphiDtphi+=(tphi(j, xi_ref_GLquad_t[q][0])*Dtphi(i, xi_ref_GLquad_t[q][0])*w);  
  }
  return tphiDtphi;
}

Real AmrDG::RefMat_phiDphi(int i,int j, int dim) const
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
  int N = qMp_1d;
  amrex::Real w;
  amrex::Real sum=0.0;
  for(int q=0; q<(int)std::pow(N,AMREX_SPACEDIM);++q){  
    w = 1.0;
    amrex::Real phi = 1.0;
    for  (int d = 0; d < AMREX_SPACEDIM; ++d){
      phi*=std::legendre(mat_idx_st[j][d], xi_ref_GLquad_s[q][d]);
    }
    
    amrex::Real dphi = 1.0;
    for  (int a = 0; a < AMREX_SPACEDIM; ++a){
      if(a!=dim)
      {
        dphi*=std::legendre(mat_idx_st[i][a], xi_ref_GLquad_s[q][a]);
      }
      else
      {
        dphi*=(std::assoc_legendre(mat_idx_st[i][dim],1,xi_ref_GLquad_s[q][dim]))
            /(std::sqrt(1.0-std::pow(xi_ref_GLquad_s[q][dim],2.0)));
      }   
    }
    
    for(int d=0; d<AMREX_SPACEDIM; ++d){
      w*=2.0/std::pow(std::assoc_legendre(N,1,xi_ref_GLquad_s[q][d]),2.0);
    }
    sum+=(phi*dphi*w);   
  }

  return sum; 
  //
}

int AmrDG::KroneckerDelta(int a, int b) const
{
  int k;
  if(a==b){k=1;}
  else{k=0;}
  return k;
}

Real AmrDG::Coefficient_c(int k,int l) const
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

//construct Vandermonde matrix for the modified basis function, 
//used in ADER predictor computation
void AmrDG::VandermondeMat()
{
  for(int i=0; i<qMp; ++i){
    for(int j=0; j<mNp; ++j){
      V[i][j] =  modPhi(j, xi_ref_GLquad[i]);//modPhi_j(x_i)==V_{ij}
      //V[i][j] =  modPhi(j, xi_ref_equidistant[i]);
    }
  } 
  
  for(int i=0; i<qMpbd; ++i){
    for(int j=0; j<mNp; ++j){
    }
  }
      amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Vbd;
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> Vbdinv;  
}

void AmrDG::InvVandermondeMat()
{
  Eigen::MatrixXd V_eigen(qMp, mNp);
  for (int i = 0; i < qMp; ++i) {
    for (int j = 0; j < mNp; ++j) {
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
  
  for (int i = 0; i < mNp; ++i) {
    for (int j = 0; j < qMp; ++j) {
        Vinv[i][j] = Vinv_eigen(i, j);
    }
  }
}
*/