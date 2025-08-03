#include "AmrDG.h"


using namespace amrex;


void AmrDG::QuadratureGaussLegendre::set_number_quadpoints()
{
  qMp_1d = (numme->p+1);

  qMp_s = (int)std::pow(qMp_1d,AMREX_SPACEDIM);//space, i.e L2 projection

  qMp_t = qMp_1d;

  qMp_st = (int)std::pow(qMp_1d,AMREX_SPACEDIM+1);//space+time

  qMp_st_bd = (int)std::pow(qMp_1d,AMREX_SPACEDIM);//(space-1)+time  
}

void AmrDG::QuadratureGaussLegendre::set_quadpoints()
{
  //Generate qMp_1d==p+1 quadrature points
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
        xi_ref_quad_st[t+N*i][0]=GLquadpts[i];
        xi_ref_quad_st[t+N*i][1]=GLquadpts[t];
      }
    }
    
    for(int t=0; t<N;++t){
      xi_ref_quad_st_bdm[0][t][0] = -bd_val;
      xi_ref_quad_st_bdm[0][t][1] = GLquadpts[t];
      xi_ref_quad_st_bdp[0][t][0] = bd_val;
      xi_ref_quad_st_bdp[0][t][1] = GLquadpts[t];
    }

    for(int i=0; i<N;++i){     
      xi_ref_quad_t[i][0]=GLquadpts[i];
      xi_ref_quad_s[i][0]=GLquadpts[i];
    } 

  #elif (AMREX_SPACEDIM == 2)
    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
        for(int t=0; t<N;++t){
          xi_ref_quad_st[t+j*N+N*N*i][0]=GLquadpts[i];
          xi_ref_quad_st[t+j*N+N*N*i][1]=GLquadpts[j]; 
          xi_ref_quad_st[t+j*N+N*N*i][2]=GLquadpts[t];
        }
      }
    }
    for(int i=0; i<N;++i){
      for(int t=0; t<N;++t){
        for(int d=0 ; d<AMREX_SPACEDIM; ++d){
          if(d==0)
          {
            xi_ref_quad_st_bdm[d][t+N*i][0] = -bd_val;
            xi_ref_quad_st_bdm[d][t+N*i][1] = GLquadpts[i]; 
            xi_ref_quad_st_bdm[d][t+N*i][2] = GLquadpts[t]; 
            xi_ref_quad_st_bdp[d][t+N*i][0] = bd_val;
            xi_ref_quad_st_bdp[d][t+N*i][1] = GLquadpts[i]; 
            xi_ref_quad_st_bdp[d][t+N*i][2] = GLquadpts[t]; 
          }
          else if(d==1)
          {
            xi_ref_quad_st_bdm[d][t+N*i][0] = GLquadpts[i];
            xi_ref_quad_st_bdm[d][t+N*i][1] = -bd_val; 
            xi_ref_quad_st_bdm[d][t+N*i][2] = GLquadpts[t]; 
            xi_ref_quad_st_bdp[d][t+N*i][0] = GLquadpts[i];
            xi_ref_quad_st_bdp[d][t+N*i][1] = bd_val; 
            xi_ref_quad_st_bdp[d][t+N*i][2] = GLquadpts[t]; 
          }
        } 
      }
    }

    for(int i=0; i<N;++i){
      xi_ref_quad_t[i][0]=GLquadpts[i];
      for(int j=0; j<N;++j){
        xi_ref_quad_s[j+N*i][0]=GLquadpts[i];
        xi_ref_quad_s[j+N*i][1]=GLquadpts[j]; 
      }
    } 
#elif (AMREX_SPACEDIM == 3)
    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
        for(int k=0; k<N;++k){
          for(int t=0; t<N;++t){
            xi_ref_quad_st[t+k*N+N*N*j+N*N*N*i][0]=GLquadpts[i];
            xi_ref_quad_st[t+k*N+N*N*j+N*N*N*i][1]=GLquadpts[j]; 
            xi_ref_quad_st[t+k*N+N*N*j+N*N*N*i][2]=GLquadpts[k]; 
            xi_ref_quad_st[t+k*N+N*N*j+N*N*N*i][3]=GLquadpts[t]; 
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
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][0] = -bd_val;
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][1] = GLquadpts[i]; 
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][2] = GLquadpts[j]; 
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][0] = bd_val;
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][1] = GLquadpts[i]; 
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][2] = GLquadpts[j];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
            }
            else if(d == 1)
            {
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][0] = GLquadpts[i];
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][1] = -bd_val; 
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][2] = GLquadpts[j]; 
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][0] = GLquadpts[i];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][1] = bd_val; 
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][2] = GLquadpts[j]; 
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
            }
            else if(d == 2)
            {
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][0] = GLquadpts[i];
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][1] = GLquadpts[j]; 
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][2] = -bd_val; 
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][0] = GLquadpts[i];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][1] = GLquadpts[j]; 
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][2] = bd_val; 
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][3] = GLquadpts[t]; 
            }
          }
        }
      }
    }

    for(int i=0; i<N;++i){
      xi_ref_quad_t[i][0]=GLquadpts[i];
      for(int j=0; j<N;++j){
        for(int k=0; k<N;++k){
          xi_ref_quad_s[k+N*j+N*N*i][0]=GLquadpts[i];
          xi_ref_quad_s[k+N*j+N*N*i][1]=GLquadpts[j]; 
          xi_ref_quad_s[k+N*j+N*N*i][2]=GLquadpts[k]; 
        }
      }
    }
#endif 

  for  (int d = 0; d < AMREX_SPACEDIM; ++d){
    xi_ref_quad_s_center[0][d]=0.0;
  }
  
}

void AmrDG::QuadratureGaussLegendre::NewtonRhapson(amrex::Real& x, int n)
{
  int niter = 1000;
    
  amrex::Real TOL= 1e-15;
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

 void AmrDG::QuadratureGaussLegendre::NewtonRhapson(amrex::Real& x, int n)
{
    int niter = 1000; // A million is excessive; it should converge much faster.
    amrex::Real TOL = 1e-15;
    amrex::Real error;
    amrex::Real x_new = x; // Start with the initial guess

    for (int it = 0; it < niter; ++it)
    {
        // Robust derivative calculation
        amrex::Real Pn_val = std::legendre(n, x);
        amrex::Real Pn_minus_1_val = std::legendre(n - 1, x);
        amrex::Real df = (n / (1.0 - x*x)) * (Pn_minus_1_val - x * Pn_val);

        // Standard update
        x_new = x - (Pn_val / df);
        error = std::abs(x_new - x);
        x = x_new;

        if (error <= TOL) { break; }
    }
}
