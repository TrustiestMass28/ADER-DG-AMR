#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

#include "AmrDG.h"


using namespace amrex;

/*
////////////////////////////////////////////////////////
QUADRATURE

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


*/