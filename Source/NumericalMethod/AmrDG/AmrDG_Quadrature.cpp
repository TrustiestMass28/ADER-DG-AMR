#include "AmrDG.h"


using namespace amrex;


template<int P>
void AmrDG::QuadratureGaussLegendre<P>::set_number_quadpoints()
{
  qMp_1d = N;

  qMp_s = (int)std::pow(N,AMREX_SPACEDIM);//space  L2 projection

  qMp_s_bd = (int)std::pow(N,AMREX_SPACEDIM-1);//surface  L2 projection

  qMp_t = N;

  qMp_st = (int)std::pow(N,AMREX_SPACEDIM+1);//space+time

  qMp_st_bd = (int)std::pow(N,AMREX_SPACEDIM);//(space-1)+time
}

template<int P>
void AmrDG::QuadratureGaussLegendre<P>::set_quadpoints()
{
  //Resize data structures holding quadrature data
  xi_ref_quad_s.resize(qMp_s,amrex::Vector<amrex::Real> (AMREX_SPACEDIM));

  xi_ref_quad_s_bdm.resize(AMREX_SPACEDIM,
                            amrex::Vector<amrex::Vector<amrex::Real>> (qMp_s_bd,
                            amrex::Vector<amrex::Real> (AMREX_SPACEDIM)));

  xi_ref_quad_s_bdp.resize(AMREX_SPACEDIM,
                            amrex::Vector<amrex::Vector<amrex::Real>> (qMp_s_bd,
                            amrex::Vector<amrex::Real> (AMREX_SPACEDIM)));

  xi_ref_quad_t.resize(qMp_t,amrex::Vector<amrex::Real> (1));

  xi_ref_quad_st.resize(qMp_st,amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1));

  xi_ref_quad_st_bdm.resize(AMREX_SPACEDIM,
                            amrex::Vector<amrex::Vector<amrex::Real>> (qMp_st_bd,
                            amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1)));

  xi_ref_quad_st_bdp.resize(AMREX_SPACEDIM,
                            amrex::Vector<amrex::Vector<amrex::Real>> (qMp_st_bd,
                            amrex::Vector<amrex::Real> (AMREX_SPACEDIM+1)));

  //construct a center point of a [-1,1]^D cell
  xi_ref_quad_s_center.resize(1,amrex::Vector<amrex::Real> (AMREX_SPACEDIM));

  //Fill quadrature points from compile-time nodes
  amrex::Real bd_val = 1.0;

  #if (AMREX_SPACEDIM == 1)
    for(int i=0; i<N;++i){
      for(int t=0; t<N;++t){
        xi_ref_quad_st[t+N*i][0]=nodes[i];
        xi_ref_quad_st[t+N*i][1]=nodes[t];
      }
    }

    xi_ref_quad_s_bdm[0][0][0] = -bd_val;
    xi_ref_quad_s_bdm[0][0][1] = nodes[0];
    xi_ref_quad_s_bdp[0][0][0] = bd_val;
    xi_ref_quad_s_bdp[0][0][1] = nodes[0];

    for(int i=0; i<N;++i){
      xi_ref_quad_t[i][0]=nodes[i];
      xi_ref_quad_s[i][0]=nodes[i];
    }

  #elif (AMREX_SPACEDIM == 2)
    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
        for(int t=0; t<N;++t){
          xi_ref_quad_st[t+j*N+N*N*i][0]=nodes[i];
          xi_ref_quad_st[t+j*N+N*N*i][1]=nodes[j];
          xi_ref_quad_st[t+j*N+N*N*i][2]=nodes[t];
        }
      }
    }

    for(int i=0; i<N;++i){
      for(int t=0; t<N;++t){
        for(int d=0 ; d<AMREX_SPACEDIM; ++d){
          if(d==0)
          {
            xi_ref_quad_st_bdm[d][t+N*i][0] = -bd_val;
            xi_ref_quad_st_bdm[d][t+N*i][1] = nodes[i];
            xi_ref_quad_st_bdm[d][t+N*i][2] = nodes[t];
            xi_ref_quad_st_bdp[d][t+N*i][0] = bd_val;
            xi_ref_quad_st_bdp[d][t+N*i][1] = nodes[i];
            xi_ref_quad_st_bdp[d][t+N*i][2] = nodes[t];
          }
          else if(d==1)
          {
            xi_ref_quad_st_bdm[d][t+N*i][0] = nodes[i];
            xi_ref_quad_st_bdm[d][t+N*i][1] = -bd_val;
            xi_ref_quad_st_bdm[d][t+N*i][2] = nodes[t];
            xi_ref_quad_st_bdp[d][t+N*i][0] = nodes[i];
            xi_ref_quad_st_bdp[d][t+N*i][1] = bd_val;
            xi_ref_quad_st_bdp[d][t+N*i][2] = nodes[t];
          }
        }
      }
    }

    for(int i=0; i<N;++i){
      for(int d=0 ; d<AMREX_SPACEDIM; ++d){
        if(d==0)
        {
          xi_ref_quad_s_bdm[d][i][0] = -bd_val;
          xi_ref_quad_s_bdm[d][i][1] = nodes[i];
          xi_ref_quad_s_bdp[d][i][0] = bd_val;
          xi_ref_quad_s_bdp[d][i][1] = nodes[i];
        }
        else if(d==1)
        {
          xi_ref_quad_s_bdm[d][i][0] = nodes[i];
          xi_ref_quad_s_bdm[d][i][1] = -bd_val;
          xi_ref_quad_s_bdp[d][i][0] = nodes[i];
          xi_ref_quad_s_bdp[d][i][1] = bd_val;
        }
      }
    }

    for(int i=0; i<N;++i){
      xi_ref_quad_t[i][0]=nodes[i];
      for(int j=0; j<N;++j){
        xi_ref_quad_s[j+N*i][0]=nodes[i];
        xi_ref_quad_s[j+N*i][1]=nodes[j];
      }
    }
#elif (AMREX_SPACEDIM == 3)
    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
        for(int k=0; k<N;++k){
          for(int t=0; t<N;++t){
            xi_ref_quad_st[t+k*N+N*N*j+N*N*N*i][0]=nodes[i];
            xi_ref_quad_st[t+k*N+N*N*j+N*N*N*i][1]=nodes[j];
            xi_ref_quad_st[t+k*N+N*N*j+N*N*N*i][2]=nodes[k];
            xi_ref_quad_st[t+k*N+N*N*j+N*N*N*i][3]=nodes[t];
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
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][1] = nodes[i];
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][2] = nodes[j];
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][3] = nodes[t];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][0] = bd_val;
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][1] = nodes[i];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][2] = nodes[j];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][3] = nodes[t];
            }
            else if(d == 1)
            {
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][0] = nodes[i];
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][1] = -bd_val;
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][2] = nodes[j];
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][3] = nodes[t];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][0] = nodes[i];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][1] = bd_val;
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][2] = nodes[j];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][3] = nodes[t];
            }
            else if(d == 2)
            {
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][0] = nodes[i];
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][1] = nodes[j];
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][2] = -bd_val;
              xi_ref_quad_st_bdm[d][t+j*N+N*N*i][3] = nodes[t];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][0] = nodes[i];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][1] = nodes[j];
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][2] = bd_val;
              xi_ref_quad_st_bdp[d][t+j*N+N*N*i][3] = nodes[t];
            }
          }
        }
      }
    }

    for(int i=0; i<N;++i){
      for(int j=0; j<N;++j){
        for(int d=0 ; d<AMREX_SPACEDIM; ++d){
          if(d == 0)
          {
            xi_ref_quad_s_bdm[d][j+N*i][0] = -bd_val;
            xi_ref_quad_s_bdm[d][j+N*i][1] = nodes[i];
            xi_ref_quad_s_bdm[d][j+N*i][2] = nodes[j];

            xi_ref_quad_s_bdp[d][j+N*i][0] = bd_val;
            xi_ref_quad_s_bdp[d][j+N*i][1] = nodes[i];
            xi_ref_quad_s_bdp[d][j+N*i][2] = nodes[j];

          }
          else if(d == 1)
          {
            xi_ref_quad_s_bdm[d][j+N*i][0] = nodes[i];
            xi_ref_quad_s_bdm[d][j+N*i][1] = -bd_val;
            xi_ref_quad_s_bdm[d][j+N*i][2] = nodes[j];

            xi_ref_quad_s_bdp[d][j+N*i][0] = nodes[i];
            xi_ref_quad_s_bdp[d][j+N*i][1] = bd_val;
            xi_ref_quad_s_bdp[d][j+N*i][2] = nodes[j];

          }
          else if(d == 2)
          {
            xi_ref_quad_s_bdm[d][j+N*i][0] = nodes[i];
            xi_ref_quad_s_bdm[d][j+N*i][1] = nodes[j];
            xi_ref_quad_s_bdm[d][j+N*i][2] = -bd_val;

            xi_ref_quad_s_bdp[d][j+N*i][0] = nodes[i];
            xi_ref_quad_s_bdp[d][j+N*i][1] = nodes[j];
            xi_ref_quad_s_bdp[d][j+N*i][2] = bd_val;

          }
        }
      }
    }

    for(int i=0; i<N;++i){
      xi_ref_quad_t[i][0]=nodes[i];
      for(int j=0; j<N;++j){
        for(int k=0; k<N;++k){
          xi_ref_quad_s[k+N*j+N*N*i][0]=nodes[i];
          xi_ref_quad_s[k+N*j+N*N*i][1]=nodes[j];
          xi_ref_quad_s[k+N*j+N*N*i][2]=nodes[k];
        }
      }
    }
#endif

  for  (int d = 0; d < AMREX_SPACEDIM; ++d){
    xi_ref_quad_s_center[0][d]=0.0;
  }

}

void AmrDG::NewtonRhapson(amrex::Real& x, int n)
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

// Explicit instantiations
template struct AmrDG::QuadratureGaussLegendre<1>;
template struct AmrDG::QuadratureGaussLegendre<2>;
template struct AmrDG::QuadratureGaussLegendre<3>;
template struct AmrDG::QuadratureGaussLegendre<4>;
template struct AmrDG::QuadratureGaussLegendre<5>;
template struct AmrDG::QuadratureGaussLegendre<6>;
template struct AmrDG::QuadratureGaussLegendre<7>;
template struct AmrDG::QuadratureGaussLegendre<8>;
template struct AmrDG::QuadratureGaussLegendre<9>;
template struct AmrDG::QuadratureGaussLegendre<10>;
