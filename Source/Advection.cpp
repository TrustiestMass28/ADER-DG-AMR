#include "Advection.h"
#include "AmrDG.h"

Advection::Advection(Simulation* _adv_sim, 
                        std::string _adv_test_case, 
                        std::string _adv_equation_type,
                        bool _adv_flag_angular_momentum,
                        bool _adv_flag_source_term)
{
  int _Q_model,_Q_model_unique;
  #if(AMREX_SPACEDIM ==2)
  {
    //base case
    _Q_model = 2;
    _Q_model_unique = _Q_model;
    
    //NB: depending on system solved, might need to modify Q_model_unique
  }

  sim=_adv_sim;
  Q_model=_Q_model;
  Q_model_unique =_Q_model_unique;
  test_case = _adv_test_case;
  equation_type =  _adv_equation_type;
  flag_angular_momentum = _adv_flag_angular_momentum;
  flag_source_term = _adv_flag_source_term;   
  
  #endif
}
amrex::Real Advection::pde_flux(int lev, int d, int q, int m, int i, int j, int k, 
                                 amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                 amrex::Vector<amrex::Real> xi)  const
{
  amrex::Real f;
  #if(AMREX_SPACEDIM ==2)  
  if(q==0){
    if(d==0){f= ((*u)[q])(i,j,k,m);}
    else if(d==1){f= 0.0;}
  }
  else if(q==1){
    if(d==0){f= ((*u)[q])(i,j,k,m);}
    else if(d==1){f= 0.0;}
  }  
  #endif  
  return f;
}

amrex::Real Advection::pde_dflux(int lev, int d, int q, int m, int i, int j, int k, 
                                 amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                 amrex::Vector<amrex::Real> xi) const
{
  amrex::Real df;
  #if(AMREX_SPACEDIM ==2) 
    if(q==0){
      if(d ==0){df= 1.0;}  
      else if(d==1){df = 0.0;}
    }
    else if(q==1){
      if(d ==0){df= 1.0;}  
      else if(d==1){df = 0.0;}
    }  
  #endif 
  return df;
}

amrex::Real Advection::pde_source(int lev, int q, int m, int i, int j, int k, 
                                      amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                      amrex::Vector<amrex::Real> xi) const
{
  amrex::Real s;
  s = 0;
  return s;
}

amrex::Real Advection::pde_IC(int lev, int q, int i,int j,int k,
                               amrex::Vector<amrex::Real> xi)
{
  const auto prob_lo = sim->dg_sim->Geom(lev).ProbLoArray();
  const auto dx     = sim->dg_sim->Geom(lev).CellSizeArray();
  
  //choose center of shape
  amrex::Vector<amrex::Real> ctr_ptr = {AMREX_D_DECL(1.0,1.0,1.0)};
  amrex::Real uw_ic;
  
  #if(AMREX_SPACEDIM ==2) 
    amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];
    amrex::Real yc = prob_lo[1] + (j+0.5) * dx[1];

    amrex::Real x_shape_ctr = ctr_ptr[0];
    amrex::Real y_shape_ctr = ctr_ptr[1];   

    amrex::Real x = (dx[0]/2.0)*xi[0]+xc; 
    amrex::Real y = (dx[1]/2.0)*xi[1]+yc;

    amrex::Real r = std::sqrt(((x-x_shape_ctr)*(x-x_shape_ctr)+(y-y_shape_ctr)*(y-y_shape_ctr)));
    if(test_case == "gaussian_shape") 
    { 
      //uw_ic= Real(1.) + (amrex::Real)std::exp(-(amrex::Real)std::pow(r,2.0)/0.01);
      //uw_ic= Real(1.) + (amrex::Real)std::exp(-(amrex::Real)std::pow(r,2.0)/0.1);
      uw_ic= std::pow(r,2.0);
    }
    else if(test_case == "discontinuous_gaussian_shape") 
    { 
      //one cell wide at resolution 32
      if((x>=1.03125) || (x<=0.96875)){uw_ic= Real(1.) + (amrex::Real)std::exp(-(amrex::Real)std::pow(r,2.0)/0.3);}
      else{uw_ic= 2.0;}     
    }      
    else if(test_case == "test") 
    { 
      uw_ic = 2.0+3.0*x;
    }
  #endif
  return uw_ic;
}

amrex::Vector<amrex::Vector<amrex::Real>> 
Advection::pde_EV_Rmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4<const amrex::Real>>* u)  const
{
  return {{1.0}}; 
}

amrex::Vector<amrex::Vector<amrex::Real>> 
Advection::pde_EV_Lmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4<const amrex::Real>>* u) const
{
  return {{1.0}}; 
} 
 //R,L matrices for limiting                                 
amrex::Vector<amrex::Vector<amrex::Real>> 
Advection::pde_EV_Rmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4< amrex::Real>>* u) const
{
  return {{1.0}}; 
}                                                                               
amrex::Vector<amrex::Vector<amrex::Real>> 
Advection::pde_EV_Lmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4<amrex::Real>>* u) const
{
  return {{1.0}}; 
}

amrex::Real Advection::pde_CFL(int d,int m,int i, int j, int k,
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u) const
{
  amrex::Real lambda;
  lambda = 1.0;
  return lambda;
} 

amrex::Real Advection::pde_BC_gDirichlet(int d, int side, int q) const
{
  //side: low=-1,high=1
  //d : dimension in which we apply BC, then we chose if we are on which side. q is solution vector component
  amrex::Real g;

  if(side == -1)
  {
    g= 0.0;
  }
  else if(side == 1)
  {
    g= 0.0;
  }
  return g;
}

amrex::Real Advection::pde_BC_gNeumann(int d, int side, int q) const
{
  //side: low=-1,high=1
  amrex::Real g;
  if(side == -1)
  {
    g= 0.0;
  }
  else if(side == 1)
  {
    g= 0.0;
  }
  return g;
}


void Advection::pde_derived_qty(int lev, int q, int m, int i, int j, int k, 
                                      amrex::Vector<amrex::Array4<amrex::Real>>* u,
                                      amrex::Vector<amrex::Real> xi) 
{}

amrex::Real Advection::pde_conservation(int lev,int d, int q,int i,int j,int k,
                                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                          amrex::Vector<amrex::Array4<amrex::Real>>* u) const
{return 0.0;}
