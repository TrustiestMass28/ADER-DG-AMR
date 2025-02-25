#include "AmrDG.h"


using namespace amrex;

void AmrDG::BasisLegendre::set_number_basis()
{
  Np_s = (int)((amrex::Real)factorial(numme->p+AMREX_SPACEDIM)
        /((amrex::Real)factorial(numme->p)*(amrex::Real)factorial(AMREX_SPACEDIM)));

  Np_st = (int)((amrex::Real)factorial(numme->p+AMREX_SPACEDIM+1)
        /((amrex::Real)factorial(numme->p)*(amrex::Real)factorial(AMREX_SPACEDIM+1)));

  //Np_t = Np_st-Np_s; not needed for DG
}

//Generate index mapping between basis fuction idx and its componetns individual idxs
void AmrDG::BasisLegendre::set_idx_mapping_s()
{
  #if (AMREX_SPACEDIM == 1)
  int ctr = 0;
  for(int ii=0; ii<=numme->p;++ii){
    basis_idx_s[ctr][0] = ii;
    if(ii==1){basis_idx_linear.push_back(ctr);} 
    ctr+=1;
  }
  #elif (AMREX_SPACEDIM == 2)
  int ctr = 0;
  for(int ii=0; ii<=numme->p;++ii){
    for(int jj=0; jj<=numme->p-ii;++jj){
      basis_idx_s[ctr][0] = ii;
      basis_idx_s[ctr][1] = jj;
      if((ii==1 && jj==0) || (ii==0 && jj==1)){basis_idx_linear.push_back(ctr);} 
      ctr+=1;      
    }
  }
  #elif (AMREX_SPACEDIM == 3)
  int ctr = 0;
  for(int ii=0; ii<=numme->p;++ii){
    for(int jj=0; jj<=numme->p-ii;++jj){
      for(int kk=0; kk<=numme->p-ii-jj;++kk){
        basis_idx_s[ctr][0] = ii;
        basis_idx_s[ctr][1] = jj;
        basis_idx_s[ctr][2] = kk;
        if((ii==1 && jj==0 && kk==0) || (ii==0 && jj==1 && kk==0) 
        || (ii==0 && jj==0 && kk==1)){basis_idx_linear.push_back(ctr);}      
        ctr+=1;
      }
    }
  }
  #endif
}

//Generate index mapping between modified basis fuction idx and its componetns 
//individual idxs
void AmrDG::BasisLegendre::set_idx_mapping_st()
{
  #if (AMREX_SPACEDIM == 1)
  int ctr = 0;
  for(int tt=0; tt<=numme->p;++tt){
    for(int ii=0; ii<=numme->p-tt;++ii){
      basis_idx_st[ctr][0] = ii;
      basis_idx_st[ctr][1] = tt;
      ctr+=1;      
    }
  }
  #elif (AMREX_SPACEDIM == 2)
  int ctr = 0;
  for(int tt=0; tt<=numme->p;++tt){
    for(int ii=0; ii<=numme->p-tt;++ii){ 
      for(int jj=0; jj<=numme->p-tt-ii;++jj){      
        basis_idx_st[ctr][0] = ii;
        basis_idx_st[ctr][1] = jj;
        basis_idx_st[ctr][2] = tt;
        ctr+=1;
      }
    }
  }
  #elif (AMREX_SPACEDIM == 3)
  int ctr = 0;
  for(int tt=0; tt<=numme->p;++tt){
    for(int ii=0; ii<=numme->p-tt;++ii){
      for(int jj=0; jj<=numme->p-tt-ii;++jj){
        for(int kk=0; kk<=numme->p-tt-ii-jj;++kk){
          basis_idx_st[ctr][0] = ii;
          basis_idx_st[ctr][1] = jj;
          basis_idx_st[ctr][2] = kk;
          basis_idx_st[ctr][3] = tt;    
          ctr+=1;
        }
      }
    }
  }
  #endif 
}

//TODO:: pass pointers to idx_map and x or them directly?
//spatial basis function, evaluated at x\in [-1,1]^{D}
amrex::Real AmrDG::BasisLegendre::phi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map, 
                                        const amrex::Vector<amrex::Real>& x) const 
{
  amrex::Real phi = 1.0;
  for  (int d = 0; d < AMREX_SPACEDIM; ++d){
    phi*=std::legendre(idx_map[idx][d], x[d]);
  }
  return phi;
}

//spatial basis function first derivative in direction d, evaluated at x\in [-1,1]^{D}
amrex::Real AmrDG::BasisLegendre::dphi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                          const amrex::Vector<amrex::Real>& x, int d) const 
{
  amrex::Real phi = 1.0;
  for  (int a = 0; a < AMREX_SPACEDIM; ++a){
    if(a!=d)
    {
      phi*=std::legendre(idx_map[idx][a], x[a]);
    }
    else
    {
      phi*=(std::assoc_legendre(idx_map[idx][d],1,x[d]))/(std::sqrt(1.0-std::pow(x[d],2.0)));
    }   
  }
  return phi;
}

//spatial basis function second derivative in direction d1 and d2, evaluated at 
//x\in [-1,1]^{D}
amrex::Real AmrDG::BasisLegendre::ddphi_s(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                            const amrex::Vector<amrex::Real>& x, int d1, int d2) const 
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
        phi*=std::legendre(idx_map[idx][a], x[a]);
      }
      else if(a==d1)
      {
        phi*=(std::assoc_legendre(idx_map[idx][d1],1,x[d1]))
            /(std::sqrt(1.0-std::pow(x[d1],2.0)));
      }
      else if(a==d2)
      {
        phi*=(std::assoc_legendre(idx_map[idx][d2],1,x[d2]))
            /(std::sqrt(1.0-std::pow(x[d2],2.0)));
      }
    }
    else
    {
      if(a!=d1 && a!=d2)
      {
        phi*=std::legendre(idx_map[idx][a], x[a]);
      }
      else
      { 
        phi*=(std::assoc_legendre(idx_map[idx][d2],2,x[d2]))
            /(1.0-std::pow(x[d2],2.0));      
      }
    }
  }
  return phi;
}

//temporal basis function
amrex::Real AmrDG::BasisLegendre::phi_t(int tidx, amrex::Real tau) const 
{
  return std::legendre(basis_idx_st[tidx][AMREX_SPACEDIM], tau);
  //TODO: use basis_idx_t?? with only 1 dimension?
}

//derivative of temporal basis function
amrex::Real AmrDG::BasisLegendre::dtphi_t(int tidx, amrex::Real tau) const
{
  return (std::assoc_legendre(basis_idx_st[tidx][AMREX_SPACEDIM],1,tau))
          /(std::sqrt(1.0-std::pow(tau,2))); 
}


//spatio temporal basis function, evaluated at x\in [-1,1]^{D+1}
amrex::Real AmrDG::BasisLegendre::phi_st(int idx, const amrex::Vector<amrex::Vector<int>>& idx_map,
                                          const amrex::Vector<amrex::Real>& x) const 
{
  amrex::Real mphi = phi_s(idx,basis_idx_st,x);
  mphi*=phi_t(idx,x[AMREX_SPACEDIM]);
  
  return mphi;
}

int AmrDG::BasisLegendre::factorial(int n)  const
{
  if (n == 0 || n == 1){
    return 1;
  } 
  else{
    return n * factorial(n - 1);
  }
}



















