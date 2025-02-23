#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

#include "AmrDG.h"


using namespace amrex;

/*
void AmrDG::BasisLegendre::set_number_basis()
{
  Print() << "HERE   "<<"\n";
}*/
/*
void AmrDG::number_modes()
{
  Np = (int)((amrex::Real)factorial(p+AMREX_SPACEDIM)
        /((amrex::Real)factorial(p)*(amrex::Real)factorial(AMREX_SPACEDIM)));
  mNp = (int)((amrex::Real)factorial(p+AMREX_SPACEDIM+1)
        /((amrex::Real)factorial(p)*(amrex::Real)factorial(AMREX_SPACEDIM+1)));
}

void AmrDG::number_quadintpts()
{
  qMp_1d = (p+1);
  qMp = (int)std::pow(qMp_1d,AMREX_SPACEDIM+1);//space+time
  qMpbd = (int)std::pow(qMp_1d,AMREX_SPACEDIM);//(space-1)+time
  qMp_L2proj = (int)std::pow(qMp_1d,AMREX_SPACEDIM);//space
}

//Generate 1D quadrature points (depending on how close we are to the boundary, 
//a different generation technique is used) //and Gauss-Legendre Quadrature points
void AmrDG::GenerateQuadPts()
{
  int N = qMp_1d;

  amrex::Vector<amrex::Real> GLquadpts;
  amrex::Real xiq = 0.0;
  amrex::Real theta = 0.0;
  for(int i=1; i<= (int)(N/2); ++i)
  {
    theta = M_PI*(i - 0.25)/((double)N + 0.5);
    
    if((1<=i) && (i<= (int)((1.0/3.0)*(double)N))){
      xiq = (1-0.125*(1.0/std::pow(N,2))+0.125*(1.0/std::pow(N,3))-(1.0/384.0)
            *(1.0/std::pow(N,4))*(39.0-28.0*(1.0/std::pow(std::sin(theta),2))))
            *std::cos(theta);
    }
    else if((i>(int)((1.0/3.0)*(double)N)) && (i<= (int)((double)N/2))){
      xiq = (1.0-(1.0/(8.0*std::pow((double)N,2)))
            +(1.0/(8.0*std::pow((double)N,3))))*std::cos(theta);
    }
    
    NewtonRhapson(xiq, N);
    GLquadpts.push_back(xiq);   
    GLquadpts.push_back(-xiq);  
  }
 
  if(N%2!=0)//if odd number, then i=1,...,N/2 will miss one value
  {
    GLquadpts.push_back(0.0);  
  }
    
  amrex::Real bd_val = 1.0;

  #if (AMREX_SPACEDIM == 1)
    for(int i=0; i<N;++i){
      for(int t=0; t<N;++t){
        xi_ref_GLquad[t+N*i][0]=GLquadpts[i];
        xi_ref_GLquad[t+N*i][1]=GLquadpts[t];
      }
    }
    
    for(int t=0; t<N;++t){
      xi_ref_GLquad_bdm[0][t][0] = -bd_val;
      xi_ref_GLquad_bdm[0][t][1] = GLquadpts[t];
      xi_ref_GLquad_bdp[0][t][0] = bd_val;
      xi_ref_GLquad_bdp[0][t][1] = GLquadpts[t];
    }

    for(int i=0; i<N;++i){     
      xi_ref_GLquad_t[i][0]=GLquadpts[i];
      xi_ref_GLquad_s[i][0]=GLquadpts[i];
      xi_ref_GLquad_L2proj[i][0]=GLquadpts[i];
    } 
  #elif (AMREX_SPACEDIM == 2)
    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
        for(int t=0; t<N;++t){
          xi_ref_GLquad[t+j*N+N*N*i][0]=GLquadpts[i];
          xi_ref_GLquad[t+j*N+N*N*i][1]=GLquadpts[j]; 
          xi_ref_GLquad[t+j*N+N*N*i][2]=GLquadpts[t];
        }
      }
    }
    for(int i=0; i<N;++i){
      for(int t=0; t<N;++t){
        for(int d=0 ; d<AMREX_SPACEDIM; ++d){
          if(d==0)
          {
            xi_ref_GLquad_bdm[d][t+N*i][0] = -bd_val;
            xi_ref_GLquad_bdm[d][t+N*i][1] = GLquadpts[i]; 
            xi_ref_GLquad_bdm[d][t+N*i][2] = GLquadpts[t]; 
            xi_ref_GLquad_bdp[d][t+N*i][0] = bd_val;
            xi_ref_GLquad_bdp[d][t+N*i][1] = GLquadpts[i]; 
            xi_ref_GLquad_bdp[d][t+N*i][2] = GLquadpts[t]; 
          }
          else if(d==1)
          {
            xi_ref_GLquad_bdm[d][t+N*i][0] = GLquadpts[i];
            xi_ref_GLquad_bdm[d][t+N*i][1] = -bd_val; 
            xi_ref_GLquad_bdm[d][t+N*i][2] = GLquadpts[t]; 
            xi_ref_GLquad_bdp[d][t+N*i][0] = GLquadpts[i];
            xi_ref_GLquad_bdp[d][t+N*i][1] = bd_val; 
            xi_ref_GLquad_bdp[d][t+N*i][2] = GLquadpts[t]; 
          }
        } 
      }
    }

    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
          xi_ref_GLquad_L2proj[j+N*i][0]=GLquadpts[i];
          xi_ref_GLquad_L2proj[j+N*i][1]=GLquadpts[j]; 
      }
    }

    for(int i=0; i<N;++i){
      xi_ref_GLquad_t[i][0]=GLquadpts[i];
      for(int j=0; j<N;++j){
          xi_ref_GLquad_s[j+N*i][0]=GLquadpts[i];
          xi_ref_GLquad_s[j+N*i][1]=GLquadpts[j]; 
      }
    } 
#elif (AMREX_SPACEDIM == 3)
    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
        for(int k=0; k<N;++k){
          for(int t=0; t<N;++t){
            xi_ref_GLquad[t+k*N+N*N*j+N*N*N*i][0]=GLquadpts[i];
            xi_ref_GLquad[t+k*N+N*N*j+N*N*N*i][1]=GLquadpts[j]; 
            xi_ref_GLquad[t+k*N+N*N*j+N*N*N*i][2]=GLquadpts[k]; 
            xi_ref_GLquad[t+k*N+N*N*j+N*N*N*i][3]=GLquadpts[t]; 
          }
        }
      }
    }
    
    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
        for(int t=0; t<N;++t){
          for(int d=0 ; d<AMREX_SPACEDIM; ++d){
            if(d == 0)
            {
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][0] = -bd_val;
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][1] = GLquadpts[i]; 
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][2] = GLquadpts[j]; 
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][0] = bd_val;
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][1] = GLquadpts[i]; 
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][2] = GLquadpts[j];
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
            }
            else if(d == 1)
            {
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][0] = GLquadpts[i];
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][1] = -bd_val; 
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][2] = GLquadpts[j]; 
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][0] = GLquadpts[i];
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][1] = bd_val; 
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][2] = GLquadpts[j]; 
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
            }
            else if(d == 2)
            {
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][0] = GLquadpts[i];
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][1] = GLquadpts[j]; 
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][2] = -bd_val; 
              xi_ref_GLquad_bdm[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][0] = GLquadpts[i];
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][1] = GLquadpts[j]; 
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][2] = bd_val; 
              xi_ref_GLquad_bdp[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
            }
          }
        }
      }
    }

    for(int i=0; i<N;++i){
      xi_ref_GLquad_t[i][0]=GLquadpts[i];
      for(int j=0; j<N;++j){
        for(int k=0; k<N;++k){
          xi_ref_GLquad_L2proj[k+N*j+N*N*i][0]=GLquadpts[i];
          xi_ref_GLquad_L2proj[k+N*j+N*N*i][1]=GLquadpts[j]; 
          xi_ref_GLquad_L2proj[k+N*j+N*N*i][2]=GLquadpts[k]; 
          
          xi_ref_GLquad_s[k+N*j+N*N*i][0]=GLquadpts[i];
          xi_ref_GLquad_s[k+N*j+N*N*i][1]=GLquadpts[j]; 
          xi_ref_GLquad_s[k+N*j+N*N*i][2]=GLquadpts[k]; 
        }
      }
    }
#endif  

}

//Generate index mapping between basis fuction idx and its componetns individual idxs
void AmrDG::PhiIdxGenerator_s()
{
  #if (AMREX_SPACEDIM == 1)
  int ctr = 0;
  for(int ii=0; ii<=p;++ii){
    mat_idx_s[ctr][0] = ii;
    if(ii==1){lin_mode_idx.push_back(ctr);} 
    ctr+=1;
  }
  #elif (AMREX_SPACEDIM == 2)
  int ctr = 0;
  for(int ii=0; ii<=p;++ii){
    for(int jj=0; jj<=p-ii;++jj){
      mat_idx_s[ctr][0] = ii;
      mat_idx_s[ctr][1] = jj;
      if((ii==1 && jj==0) || (ii==0 && jj==1)){lin_mode_idx.push_back(ctr);} 
      ctr+=1;      
    }
  }
  #elif (AMREX_SPACEDIM == 3)
  int ctr = 0;
  for(int ii=0; ii<=p;++ii){
    for(int jj=0; jj<=p-ii;++jj){
      for(int kk=0; kk<=p-ii-jj;++kk){
        mat_idx_s[ctr][0] = ii;
        mat_idx_s[ctr][1] = jj;
        mat_idx_s[ctr][2] = kk;
        if((ii==1 && jj==0 && kk==0) || (ii==0 && jj==1 && kk==0) 
        || (ii==0 && jj==0 && kk==1)){lin_mode_idx.push_back(ctr);}      
        ctr+=1;
      }
    }
  }
  #endif
}

//Generate index mapping between modified basis fuction idx and its componetns 
//individual idxs
void AmrDG::PhiIdxGenerator_st()
{
  #if (AMREX_SPACEDIM == 1)
  int ctr = 0;
  for(int tt=0; tt<=p;++tt){
    for(int ii=0; ii<=p-tt;++ii){
      mat_idx_st[ctr][0] = ii;
      mat_idx_st[ctr][1] = tt;
      ctr+=1;      
    }
  }
  #elif (AMREX_SPACEDIM == 2)
  int ctr = 0;
  for(int tt=0; tt<=p;++tt){
    for(int ii=0; ii<=p-tt;++ii){ 
      for(int jj=0; jj<=p-tt-ii;++jj){      
        mat_idx_st[ctr][0] = ii;
        mat_idx_st[ctr][1] = jj;
        mat_idx_st[ctr][2] = tt;
        ctr+=1;
      }
    }
  }
  #elif (AMREX_SPACEDIM == 3)
  int ctr = 0;
  for(int tt=0; tt<=p;++tt){
    for(int ii=0; ii<=p-tt;++ii){
      for(int jj=0; jj<=p-tt-ii;++jj){
        for(int kk=0; kk<=p-tt-ii-jj;++kk){
          mat_idx_st[ctr][0] = ii;
          mat_idx_st[ctr][1] = jj;
          mat_idx_st[ctr][2] = kk;
          mat_idx_st[ctr][3] = tt;    
          ctr+=1;
        }
      }
    }
  }
  #endif 
}

//temporal basis function
amrex::Real AmrDG::tphi(int tidx, amrex::Real tau) const
{
  return std::legendre(mat_idx_st[tidx][AMREX_SPACEDIM], tau);
}

//derivative of temporal basis function
amrex::Real AmrDG::Dtphi(int tidx, amrex::Real tau) const
{
  return (std::assoc_legendre(mat_idx_st[tidx][AMREX_SPACEDIM],1,tau))
          /(std::sqrt(1.0-std::pow(tau,2))); 
}

//spatio temporal basis function, evaluated at x\in [-1,1]^{D+1}
amrex::Real AmrDG::modPhi(int idx, amrex::Vector<amrex::Real> x) const
{
  amrex::Real mphi = 1.0;
  for  (int d = 0; d < AMREX_SPACEDIM; ++d){
    mphi*=std::legendre(mat_idx_st[idx][d], x[d]);
  }
  mphi*=tphi(idx,x[AMREX_SPACEDIM]);
  
  return mphi;
}

//spatial basis function, evaluated at x\in [-1,1]^{D}
Real AmrDG::Phi(int idx, amrex::Vector<amrex::Real> x) const
{
  amrex::Real phi = 1.0;
  for  (int d = 0; d < AMREX_SPACEDIM; ++d){
    phi*=std::legendre(mat_idx_s[idx][d], x[d]);
  }
  return phi;
}

//spatial basis function first derivative in direction d, evaluated at x\in [-1,1]^{D}
amrex::Real AmrDG::DPhi(int idx, amrex::Vector<amrex::Real> x, int d) const
{
  amrex::Real phi = 1.0;
  for  (int a = 0; a < AMREX_SPACEDIM; ++a){
    if(a!=d)
    {
      phi*=std::legendre(mat_idx_s[idx][a], x[a]);
    }
    else
    {
      phi*=(std::assoc_legendre(mat_idx_s[idx][d],1,x[d]))/(std::sqrt(1.0-std::pow(x[d],2.0)));
    }   
  }
  return phi;
}

//spatial basis function second derivative in direction d1 and d2, evaluated at 
//x\in [-1,1]^{D}
amrex::Real AmrDG::DDPhi(int idx, amrex::Vector<amrex::Real> x, int d1, int d2) const
{
  //evaluates basis function second derivatives in d1 and d2 direction at desired 
  //reference location x\in [-1,1]
  //NB: in C++ std library, when using std::assoc_legendre, the (-1)^{-m} term 
  //is not considered, therefore should not be included 
  //in reverse computations
  amrex::Real phi = 1.0;
  for  (int a = 0; a < AMREX_SPACEDIM; ++a){
  
    if(d1!=d2)
    {
      if(a!=d1 && a!=d2)
      {
        phi*=std::legendre(mat_idx_s[idx][a], x[a]);
      }
      else if(a==d1)
      {
        phi*=(std::assoc_legendre(mat_idx_s[idx][d1],1,x[d1]))
            /(std::sqrt(1.0-std::pow(x[d1],2.0)));
      }
      else if(a==d2)
      {
        phi*=(std::assoc_legendre(mat_idx_s[idx][d2],1,x[d2]))
            /(std::sqrt(1.0-std::pow(x[d2],2.0)));
      }
    }
    else
    {
      if(a!=d1 && a!=d2)
      {
        phi*=std::legendre(mat_idx_s[idx][a], x[a]);
      }
      else
      { 
        phi*=(std::assoc_legendre(mat_idx_s[idx][d2],2,x[d2]))
            /(1.0-std::pow(x[d2],2.0));      
      }
    }
  }
  return phi;
}

void AmrDG::NewtonRhapson(amrex::Real& x, int n)
{
  int niter = 10000000;
  
  amrex::Real TOL= 1e-30;
  amrex::Real error;
  amrex::Real x_new = 0.0;

  for(int it = 0; it<niter; ++it)
  {
    amrex::Real df = (std::assoc_legendre(n,1,x))/(std::sqrt(1.0-std::pow(x,2.0)));
    amrex::Real f = std::legendre(n, x);
    x_new = x-(f/df);
    
    error = std::abs(x_new-x);
    x = x_new;
    
    if(error<=TOL){break;}    
  }
}

int AmrDG::factorial(int n)  const
{
  if (n == 0 || n == 1){
    return 1;
  } 
  else{
    return n * factorial(n - 1);
  }
}
*/