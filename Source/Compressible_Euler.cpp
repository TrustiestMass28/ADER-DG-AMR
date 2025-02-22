#include "Compressible_Euler.h"

using namespace amrex;

void Compressible_Euler::settings(std::string _euler_case){

  model_case = _euler_case;

  //settings to include new terms and or equations in the system
  if(model_case == "keplerian_disc"){
    flag_angular_momentum = true;
    flag_source_term = true;
  }
  else{
    flag_angular_momentum = false;
    flag_source_term = false;
  }

  int _Q_model;
  int _Q_model_unique;

  #if(AMREX_SPACEDIM ==2)
    //base case
    _Q_model = 4;
    _Q_model_unique = _Q_model;
    
    //additional terms
    if(flag_angular_momentum){_Q_model = 5;}
    
    //NB: depending on system solved, might need to modify Q_model_unique
  #elif(AMREX_SPACEDIM ==3)
    _Q_model = 5;
    _Q_model_unique = _Q_model;
    
    //additional terms which are derived from the existing ones
    if(flag_angular_momentum){_Q_model = 8;}  
  #endif

  Q_model=_Q_model;
  Q_model_unique =_Q_model_unique;

  gamma_adiab = 1.4;
}  


/*
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


amrex::Real Compressible_Euler::pde_IC(int lev, int q, int i,int j,int k,
                                      amrex::Vector<amrex::Real> xi)
{
  const auto prob_lo = numerical_pde->Geom(lev).ProbLoArray();
  const auto dx     = numerical_pde->Geom(lev).CellSizeArray();
  
  amrex::Real uw_ic; 
#if (AMREX_SPACEDIM == 1)
  amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];

  amrex::Real x = 0.5*dx[0]*xi[0]+xc;
#elif(AMREX_SPACEDIM ==2) 
  amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];
  amrex::Real yc = prob_lo[1] + (j+0.5) * dx[1];

  amrex::Real x = (dx[0]/2.0)*xi[0]+xc; 
  amrex::Real y = (dx[1]/2.0)*xi[1]+yc;

  if(test_case == "isentropic_vortex")
  {
    //Kevin Schaal et. al., Astrophysical hydrodynamics with a high-order 
    //discontinuous Galerking scheme and adaptive mesh refinement, 
    //Royal Astronomical Society (2015) ,https://doi.org/10.1093/mnras/stv1859
    
    //shape center
    amrex::Vector<amrex::Real> ctr_ptr = {AMREX_D_DECL(5.0,5.0,5.0)};
    amrex::Real x_shape_ctr = ctr_ptr[0];
    amrex::Real y_shape_ctr = ctr_ptr[1];    
    
    amrex::Real r = std::sqrt(((x-x_shape_ctr)*(x-x_shape_ctr)+(y-y_shape_ctr)
                    *(y-y_shape_ctr)));
                                    
    amrex::Real beta= 5.0;
    amrex::Real u1_infty = 1.0;
    amrex::Real u2_infty = 0.0;    
    amrex::Real rho = std::pow(1.0-(((gamma_adiab-1.0)*std::pow(beta,2.0))
                      /(8.0*gamma_adiab*std::pow(M_PI,2)))*std::exp(1.0-std::pow(r,2.0)),
                      (1.0/(gamma_adiab-1.0)));
                      
    amrex::Real prs = std::pow(rho,gamma_adiab);
    amrex::Real du1 =-(y-y_shape_ctr)*(beta/(M_PI*2.0))
                    *std::exp((1.0-std::pow(r,2.0))/2.0);
                    
    amrex::Real du2 =(x-x_shape_ctr)*(beta/(M_PI*2.0))
                    *std::exp((1.0-std::pow(r,2.0))/2.0);
                    
    amrex::Real u1=u1_infty+du1;
    
    amrex::Real u2=u2_infty+du2;
    
    amrex::Real e=(prs/(rho*(gamma_adiab-1.0)))
                  +0.5*(std::pow(u1,2.0)+std::pow(u2,2.0));
    
    if(q==0){uw_ic= rho;}
    else if(q==1){uw_ic= rho*u1;}
    else if(q==2){uw_ic= rho*u2;}
    else if(q==3){uw_ic= rho*e;}
  }
  else if(test_case == "isentropic_vortex_static")
  {
    //Kevin Schaal et. al., Astrophysical hydrodynamics with a high-order 
    //discontinuous Galerking scheme and adaptive mesh refinement, 
    //Royal Astronomical Society (2015) ,https://doi.org/10.1093/mnras/stv1859
    
    //shape center
    amrex::Vector<amrex::Real> ctr_ptr = {AMREX_D_DECL(5.0,5.0,5.0)};
    amrex::Real x_shape_ctr = ctr_ptr[0];
    amrex::Real y_shape_ctr = ctr_ptr[1];    
    
    amrex::Real r = std::sqrt(((x-x_shape_ctr)*(x-x_shape_ctr)+(y-y_shape_ctr)
                    *(y-y_shape_ctr)));
                                    
    amrex::Real beta= 5.0;
    amrex::Real u1_infty = 0.0;
    amrex::Real u2_infty = 0.0;    
    amrex::Real rho = std::pow(1.0-(((gamma_adiab-1.0)*std::pow(beta,2.0))
                      /(8.0*gamma_adiab*std::pow(M_PI,2)))*std::exp(1.0-std::pow(r,2.0)),
                      (1.0/(gamma_adiab-1.0)));
                      
    amrex::Real prs = std::pow(rho,gamma_adiab);
    amrex::Real du1 =-(y-y_shape_ctr)*(beta/(M_PI*2.0))
                    *std::exp((1.0-std::pow(r,2.0))/2.0);
                    
    amrex::Real du2 =(x-x_shape_ctr)*(beta/(M_PI*2.0))
                    *std::exp((1.0-std::pow(r,2.0))/2.0);
                    
    amrex::Real u1=u1_infty+du1;
    
    amrex::Real u2=u2_infty+du2;
    
    amrex::Real e=(prs/(rho*(gamma_adiab-1.0)))
                  +0.5*(std::pow(u1,2.0)+std::pow(u2,2.0));
    
    if(q==0){uw_ic= rho;}
    else if(q==1){uw_ic= rho*u1;}
    else if(q==2){uw_ic= rho*u2;}
    else if(q==3){uw_ic= rho*e;}
  }
  else if(test_case == "double_mach_reflection")
  {
    gamma_adiab = 1.4;
    amrex::Real alpha=M_PI/3.0;
    amrex::Real xp = (x-1.0/6.0)*std::cos(alpha)-y*std::sin(alpha);
    
    amrex::Real prs = (xp < 0.0) ? 116.5 : 1.0; 
    amrex::Real rho = (xp < 0.0) ? 8.0 : 1.4; 
    amrex::Real u1 = (xp < 0.0) ? (8.25*std::cos(alpha)) : 0.0; 
    amrex::Real u2 = (xp < 0.0) ? (8.25*std::sin(alpha)) : 0.0;      
    amrex::Real rho_e=(prs/(gamma_adiab-1.0))+0.5*rho*(std::pow(u1,2.0)+std::pow(u2,2.0));
                  
    if(q==0){uw_ic= rho;}
    else if(q==1){uw_ic= rho*u1;}
    else if(q==2){uw_ic= rho*u2;}
    else if(q==3){uw_ic= rho_e;}    
  }
  else if(test_case == "kelvin_helmolz_instability")
  {       
    //Kevin Schaal et. al., Astrophysical hydrodynamics with a high-order
    //discontinuous Galerking scheme and adaptive mesh refinement, 
    //Royal Astronomical Society (2015) ,https://doi.org/10.1093/mnras/stv1859
    amrex::Real w0=0.1;
    amrex::Real sigma = 0.05/std::sqrt(2.0);
    
    amrex::Real rho =  ((y > 0.25) && (y < 0.75)) ? 2.0 : 1.0; 
    amrex::Real prs=2.5;
    amrex::Real u1 = ((y > 0.25) && (y < 0.75)) ? 0.5 : -0.5; 
    amrex::Real u2 = w0*std::sin(4.0*M_PI*x)*(std::exp(-(std::pow(y-0.25,2.0))
                    /(2.0*std::pow(sigma,2.0)))+std::exp(-(std::pow(y-0.75,2.0))
                    /(2.0*std::pow(sigma,2))));
                    
    amrex::Real rho_e=(prs/(gamma_adiab-1.0))+0.5*rho*(std::pow(u1,2.0)+std::pow(u2,2.0));
    
    if(q==0){uw_ic= rho;}
    else if(q==1){uw_ic= rho*u1;}
    else if(q==2){uw_ic= rho*u2;}
    else if(q==3){uw_ic= rho_e;}
  }
  else if(test_case == "richtmeyer_meshkov_instability")
  {
    //Jesse Chan et al, On the Entropy Projection and the Robustness of High 
    //Order Entropy Stable Discontinuous Galerkin Schemes for Under-Resolved
    //Flows (2022),  https://doi.org/10.3389/fphy.2022.898028
    
    amrex::Real length = 40.0;
    
    amrex::Real rho =  smooth_discontinuity(y-(18.0+2.0*std::cos(6.0*M_PI*x/length)), 
                      1.0, 1.0/4.0, 2.0)+smooth_discontinuity(std::abs(y-4.0)-2.0, 3.22, 0.0, 2.0);
                      
    amrex::Real prs =  smooth_discontinuity(std::abs(y-4.0)-2.0, 4.9, 1.0, 2.0);
    amrex::Real u1 = 0.0;
    amrex::Real u2 = 0.0;
    amrex::Real rho_e=(prs/(gamma_adiab-1.0))+0.5*rho*(std::pow(u1,2.0)+std::pow(u2,2.0));
    if(q==0){uw_ic= rho;}
    else if(q==1){uw_ic= rho*u1;}
    else if(q==2){uw_ic= rho*u2;}
    else if(q==3){uw_ic= rho_e;}
  
  }
  else if(test_case == "keplerian_disc")
  {
    //Kevin Schaal et. al., Astrophysical hydrodynamics with a 
    //high-order discontinuous Galerking scheme and adaptive mesh refinement,
    //Royal Astronomical Society (2015) ,https://doi.org/10.1093/mnras/stv1859

    amrex::Vector<amrex::Real> ctr_ptr = {AMREX_D_DECL(3.0,3.0,3.0)};
    amrex::Real x_shape_ctr = ctr_ptr[0];
    amrex::Real y_shape_ctr = ctr_ptr[1];   
  
    amrex::Real r = std::sqrt(((x-x_shape_ctr)*(x-x_shape_ctr)+(y-y_shape_ctr)
                    *(y-y_shape_ctr)));
        
    gamma_adiab = 5.0/3.0;
    
    amrex::Real p0 = 1e-5;
    amrex::Real rho0=1e-5;
    amrex::Real rhoD=1.0;
    amrex::Real Dr=0.1;
    amrex::Real xp = x-x_shape_ctr;
    amrex::Real yp = y-y_shape_ctr;
    //define bounds for rho definition, to ease the implementation
    amrex::Real b1 = 0.5-Dr/2.0;
    amrex::Real b2 = 0.5+Dr/2.0;
    amrex::Real b3 = 2.0-Dr/2.0;
    amrex::Real b4 = 2.0+Dr/2.0;
    amrex::Real v12 = ((rhoD-rho0)/Dr)*(r-(0.5-Dr/2.0))+rho0;
    amrex::Real v34 = ((rho0-rhoD)/Dr)*(r-(2.0-Dr/2.0))+rhoD;
    amrex::Real rho = (r < b1) ? rho0 : (r >=b1 && r < b2) ? v12 : 
                      (r >=b2 && r < b3) ? rhoD : (r >=b3 && r < b4) ? v34 : 
                      (r >=b4) ? rho0 : rho0; 
                      
    amrex::Real prs = p0;
    amrex::Real u1 = (r > 0.5-2.0*Dr && r < 2.0+2.0*Dr) 
                    ? (-yp)/(std::pow(r,3.0/2.0)) : 0; 
    amrex::Real u2 = (r > 0.5-2.0*Dr && r < 2.0+2.0*Dr) 
                    ? (xp)/(std::pow(r,3.0/2.0)) : 0; 
                    
    amrex::Real e=(prs/(rho*(gamma_adiab-1.0)))
          +0.5*(std::pow(u1,2.0)+std::pow(u2,2.0));
    
    if(q==0){uw_ic= rho;}
    else if(q==1){uw_ic= rho*u1;}
    else if(q==2){uw_ic= rho*u2;}
    else if(q==3){uw_ic= rho*e;}
    
    if(flag_angular_momentum)
    {
      //TODO:use xp or x as locations for Ang.mom?
      //->use x and not shifted position because in source we also use x and also in derived_qty we use x
      //outside of this IC function we dont have informations on where the shape is centered. and anyway is just 
      //a scalar shift, nothing will change in final conservation
      amrex::Real x1 = x;
      amrex::Real x2 = y;
      amrex::Real L3 = x1*rho*u2-x2*rho*u1;
      if(q==4){uw_ic= L3;}
    }   
  }
  else if(test_case == "radial_shock_tube")
  { 
    //Mishra and Hiltelbrand, Entropy stable shock capturing space-time DG 
    //scheme for systems of conservation laws (2014),https://doi.org/10.1007/s00211-013-0558-0

    amrex::Vector<amrex::Real> ctr_ptr = {AMREX_D_DECL(3.0,3.0,3.0)};
    amrex::Real x_shape_ctr = ctr_ptr[0];
    amrex::Real y_shape_ctr = ctr_ptr[1];   
  
    amrex::Real r = std::sqrt(((x-x_shape_ctr)*(x-x_shape_ctr)+(y-y_shape_ctr)
                    *(y-y_shape_ctr)));
                    
    amrex::Real rho =  (r > 0.4) ? 0.125 : 1.0; 
    amrex::Real prs =  (r > 0.4) ? 0.125 : 1.0;
    amrex::Real u1 = 0.0;
    amrex::Real u2 = 0.0;
    amrex::Real rho_e=(prs/(gamma_adiab-1.0))+0.5*rho*(std::pow(u1,2.0)+std::pow(u2,2.0));
    if(q==0){uw_ic= rho;}
    else if(q==1){uw_ic= rho*u1;}
    else if(q==2){uw_ic= rho*u2;}
    else if(q==3){uw_ic= rho_e;}
  }
#elif(AMREX_SPACEDIM ==3) 
    amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];
    amrex::Real yc = prob_lo[1] + (j+0.5) * dx[1];
    amrex::Real zc = prob_lo[2] + (k+0.5) * dx[2];
    
    amrex::Real x = 0.5*dx[0]*xi[0]+xc;
    amrex::Real y = 0.5*dx[1]*xi[1]+yc; 
    amrex::Real z = 0.5*dx[2]*xi[2]+zc; 
#endif 
  return uw_ic;
}

//For Euler equations, we specifiy Dirichlet,Neumann BCs w.r.t rho,u1,u2,p. 
////(p used to five BC to energy)
//Then these specified values are mapped (inside AmrDG::BoundaryCondition::operator(), 
////also considering the other components BCs) 
//to our solution vector components i.e to rho,rho_u1,rho_u2,rho_e

amrex::Real Compressible_Euler::pde_BC_gDirichlet(int d, int side, int q) const
{
  //side: low=-1,high=1
  //d : dimension in which we apply BC, then we chose if we are on which side. 
  //q is solution vector component
  
  amrex::Real g;
  if(test_case == "double_mach_reflection")
  {
    if(side == -1)
    {
      g= 10.06;
    }
    else if(side == 1)
    {
      g= 10.06;
    }
  }
  return g;
}

amrex::Real Compressible_Euler::pde_BC_gNeumann(int d, int side, int q) const
{
  //side: low=-1,high=1
  amrex::Real g;

  //if(euler_test_case ==  "richtmeyer_meshkov_instability")
  //{
    //wall BC
    //actually since we only have neumann in one direction, 
    //no need to specify d
    if(d==0)
    {
      if(side == -1)
      {
        g= 0.0;
      }
      else if(side == 1)
      {
        g= 0.0;
      }  
    }
    else if(d==1)
    {
      if(side == -1)
      {
        g= 0.0;
      }
      else if(side == 1)
      {
        g= 0.0;
      }  
    }
 // }
  return g;
}

amrex::Real Compressible_Euler::pde_source(int lev, int q, int m, int i, int j, int k, 
                                          amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                          amrex::Vector<amrex::Real> xi) const
{
  amrex::Real s;
  if(test_case == "keplerian_disc")
  {
    const auto prob_lo = numerical_pde->Geom(lev).ProbLoArray();
    const auto dx     = numerical_pde->Geom(lev).CellSizeArray();
  
    amrex::Vector<amrex::Real> ctr_ptr = {AMREX_D_DECL(3.0,3.0,3.0)};
    amrex::Real x_shape_ctr = ctr_ptr[0];
    amrex::Real y_shape_ctr = ctr_ptr[1];     
    
    amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];
    amrex::Real yc = prob_lo[1] + (j+0.5) * dx[1];
    amrex::Real x = (dx[0]/2.0)*xi[0]+xc; 
    amrex::Real y = (dx[1]/2.0)*xi[1]+yc;    
    
    amrex::Real xp = x-x_shape_ctr;
    amrex::Real yp = y-y_shape_ctr;

    amrex::Real r = std::sqrt((xp*xp+yp*yp)); 
     
    amrex::Real Dr = 0.1;
    amrex::Real eps = 0.25;
    amrex::Real ax =  (0.5-0.5*Dr < r) ? (-xp/std::pow(r,3.0)) 
                      : (-xp/(r*(std::pow(r,2.0)+std::pow(eps,2.0)))); 
    amrex::Real ay =  (0.5-0.5*Dr < r) ? (-yp/std::pow(r,3.0)) 
                      : (-yp/(r*(std::pow(r,2.0)+std::pow(eps,2.0)))); 
    
    if(q ==0){s= 0.0;}
    else if(q ==1){s= -((*u)[0])(i,j,k,m)*ax;}  
    else if(q ==2){s= -((*u)[0])(i,j,k,m)*ay;}  
    else if(q ==3){s= -((*u)[1])(i,j,k,m)*ax-((*u)[2])(i,j,k,m)*ay;}  
    else if(q ==4){s= 0.0;}  
  }
  return s;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void Compressible_Euler::pde_derived_qty(int lev, int q, int m, int i, int j, int k, 
                                      amrex::Vector<amrex::Array4<amrex::Real>>* u,
                                      amrex::Vector<amrex::Real> xi)
{
  //get non-unique, i.e derived, quantities point-wise values
  const auto prob_lo = numerical_pde->Geom(lev).ProbLoArray();
  const auto dx     = numerical_pde->Geom(lev).CellSizeArray();
#if(AMREX_SPACEDIM ==2)  
  amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];
  amrex::Real yc = prob_lo[1] + (j+0.5) * dx[1];

  amrex::Real x = (dx[0]/2.0)*xi[0]+xc; 
  amrex::Real y = (dx[1]/2.0)*xi[1]+yc;
  
  amrex::Real x1 = x;
  amrex::Real x2 = y;
  
  amrex::Real rho_u1 = ((*u)[1])(i,j,k,m);
  amrex::Real rho_u2 = ((*u)[2])(i,j,k,m);

  if(q==4){((*u)[4])(i,j,k,m)=x1*rho_u2-x2*rho_u1;} 
  
#elif(AMREX_SPACEDIM ==3)
  amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];
  amrex::Real yc = prob_lo[1] + (j+0.5) * dx[1];
  amrex::Real zc = prob_lo[2] + (k+0.5) * dx[2];

  amrex::Real x = (dx[0]/2.0)*xi[0]+xc; 
  amrex::Real y = (dx[1]/2.0)*xi[1]+yc;
  amrex::Real z = (dx[2]/2.0)*xi[2]+yc;
  
  amrex::Real x1 = x;
  amrex::Real x2 = y;
  amrex::Real x3 = z;
  
  amrex::Real rho_u1 = ((*u)[1])(i,j,k,m);
  amrex::Real rho_u2 = ((*u)[2])(i,j,k,m);
  amrex::Real rho_u3 = ((*u)[3])(i,j,k,m);

  if(q==5){((*u)[5])(i,j,k,m)=x2*rho_u3-x3*rho_u2;} 
  else if(q==6){((*u)[6])(i,j,k,m)=x3*rho_u1-x1*rho_u3;} 
  else if(q==7){((*u)[7])(i,j,k,m)=x1*rho_u2-x2*rho_u1;} 
  
#endif 
}
                                      
amrex::Real Compressible_Euler::pde_flux(int lev, int d, int q, int m, int i, int j, int k, 
                                        amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                        amrex::Vector<amrex::Real> xi) const
{
  amrex::Real f;
  amrex::Real prs=Pressure(u,i,j,k,m);
#if(AMREX_SPACEDIM ==2)    
    if(d==0)//f1
    {
      if(q==0){f= ((*u)[1])(i,j,k,m);}
      else if(q==1){f=(std::pow(((*u)[1])(i,j,k,m),2.0)/((*u)[0])(i,j,k,m))+prs;}
      else if(q==2){f=((*u)[1])(i,j,k,m)*((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      else if(q==3){f=(((*u)[3])(i,j,k,m)+prs)*(((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m));}
    }
    
    else if(d==1)//f2
    {
      if(q==0){f= ((*u)[2])(i,j,k,m);}
      else if(q==1){f=((*u)[2])(i,j,k,m)*((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      else if(q==2){f= (std::pow(((*u)[2])(i,j,k,m),2.0)/((*u)[0])(i,j,k,m))+prs;}
      else if(q==3){f= (((*u)[3])(i,j,k,m)+prs)*(((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m));}
    }
    
    if(flag_angular_momentum)
    { 
      //implementation of angular momentum conservation law
      const auto prob_lo = numerical_pde->Geom(lev).ProbLoArray();
      const auto prob_hi = numerical_pde->Geom(lev).ProbHiArray();
      const auto dx     = numerical_pde->Geom(lev).CellSizeArray();
      amrex::Real x1c = prob_lo[0] + (i+Real(0.5)) * dx[0];
      amrex::Real x2c = prob_lo[1] + (j+Real(0.5)) * dx[1];
      amrex::Real x1 = dx[0]*0.5*(xi[0])+x1c;
      amrex::Real x2 = dx[1]*0.5*(xi[1])+x2c;

      amrex::Real L3 = x1*((*u)[2])(i,j,k,m)-x2*((*u)[1])(i,j,k,m);
   
      if(d==0)//f1
      {
        if(q==4){f= L3*((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m)-x2*prs;}
      }
      else if(d==1)//f2
      {
        if(q==4){f= L3*((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m)+x1*prs;}
      }
    }  
#elif(AMREX_SPACEDIM ==3)
  {//q=0->rho,  q=1->rho*u1, q=2->rho*u2, q=3->rho*u3,q=4->rho*e
    if(d==0)//f1
    {
      if(q==0){f= ((*u)[1])(i,j,k,m);}
      else if(q==1){f= (std::pow(((*u)[1])(i,j,k,m),2.0)/((*u)[0])(i,j,k,m))+prs;}
      else if(q==2){f= ((*u)[1])(i,j,k,m)*((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      else if(q==3){f= ((*u)[1])(i,j,k,m)*((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      else if(q==4){f= (((*u)[4])(i,j,k,m)+prs)*(((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m));}   
    }
    
    else if(d==1)//f2
    {
      if(q==0){f= ((*u)[2])(i,j,k,m);}
      else if(q==1){f= ((*u)[2])(i,j,k,m)*((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      else if(q==2){f= (std::pow(((*u)[2])(i,j,k,m),2.0)/((*u)[0])(i,j,k,m))+prs;}
      else if(q==3){f= ((*u)[2])(i,j,k,m)*((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      else if(q==4){f= (((*u)[4])(i,j,k,m)+prs)*(((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m));}
    }
    else if(d==2)//f3
    {
      if(q==0){f= ((*u)[3])(i,j,k,m);}
      else if(q==1){f=((*u)[3])(i,j,k,m)*((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      else if(q==2){f= ((*u)[3])(i,j,k,m)*((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      else if(q==3){f= (std::pow(((*u)[3])(i,j,k,m),2.0)/((*u)[0])(i,j,k,m))+prs;}
      else if(q==4){f= (((*u)[4])(i,j,k,m)+prs)*(((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m));}
    }
    
    if(euler_flag_angular_momentum)
    { 
      //implementation of angular momentum conservation law
      const auto prob_lo = numerical_pde->Geom(lev).ProbLoArray();
      const auto prob_hi = numerical_pde->Geom(lev).ProbHiArray();
      const auto dx     = numerical_pde->Geom(lev).CellSizeArray();
      amrex::Real x1c = prob_lo[0] + (i+Real(0.5)) * dx[0];
      amrex::Real x2c = prob_lo[1] + (j+Real(0.5)) * dx[1];
      amrex::Real x3c = prob_lo[2] + (k+Real(0.5)) * dx[2];
      amrex::Real x1 = dx[0]*0.5*(xi[0])+x1c;
      amrex::Real x2 = dx[1]*0.5*(xi[1])+x2c;
      amrex::Real x3 = dx[2]*0.5*(xi[2])+x3c;
   
      amrex::Real L1 = x2*((*u)[3])(i,j,k,m)-x3*((*u)[2])(i,j,k,m);
      amrex::Real L2 = x3*((*u)[1])(i,j,k,m)-x1*((*u)[3])(i,j,k,m);
      amrex::Real L3 = x1*((*u)[2])(i,j,k,m)-x2*((*u)[1])(i,j,k,m);
      
      if(d==0)//f1
      {
        if(q==5){f= L1*((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m);}
        else if(q==6){f= L2*((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m)+x3*prs;}
        else if(q==7){f= L3*((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m)-x2*prs;}
      }
      else if(d==1)//f2
      {
        if(q==5){f= L1*((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m)-x3*prs;}
        else if(q==6){f= L2*((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m);}
        else if(q==7){f= L3*((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m)+x1*prs;}
      }
      else if(d==2)//f3
      {
        if(q==5){f= L1*((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m)+x2*prs;}
        else if(q==6){f= L2*((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m)-x1*prs;}
        else if(q==7){f= L3*((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m);}
      }
    }
  }
#endif

  return f;
} 

amrex::Real Compressible_Euler::pde_dflux(int lev, int d, int q, int m, int i, int j, int k, 
                                          amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                          amrex::Vector<amrex::Real> xi) const
{
  amrex::Real df;
  amrex::Real un;
  amrex::Real c = Soundspeed(u, i,j,k,m);
  
  amrex::Real n[AMREX_SPACEDIM];
#if(AMREX_SPACEDIM ==2) 
  if(d==0){n[0]=1.0;n[1]=0.0;}
  else if(d==1){n[0]=0.0;n[1]=1.0;}
#elif(AMREX_SPACEDIM ==3)
  if(d==0){n[0]=1.0;n[1]=0.0;n[2]=0.0;}
  else if(d==1){n[0]=0.0;n[1]=1.0;n[2]=0.0;}
  else if(d==2){n[0]=0.0;n[1]=0.0;n[2]=1.0;}
#endif

  un = 0.0;
  for(int _d=0;_d<AMREX_SPACEDIM; ++_d)
  {
    un+=((((*u)[_d+1])(i,j,k,m)/((*u)[0])(i,j,k,m))*n[_d]);
  }
  
  df=std::max({std::abs(un-c), std::abs(un), std::abs(un+c)});
  
  return df;
}

amrex::Real Compressible_Euler::get_DU_from_U_w(int d, int q, int i, int j, int k,
                                                 amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                                 amrex::Vector<amrex::Real> xi) const
{
  //get the derivative of the specified solution component along specified direction
  //used for Euler equations, divide by density ((*uw)[0])(i,j,k,m)) to obtain 
  //just velocity in case we need derivative of density, then we do not divide by it
  amrex::Real derivative_rho = 0.0;
  amrex::Real derivative = 0.0;
  amrex::Real derivative_rhou_q = 0.0;
  amrex::Real rho=0.0;
  amrex::Real rhou_q = 0.0;
  for(int m=0; m<numerical_pde->Np; ++m)
  { 
    if(q!=0){
      derivative_rho+=((*uw)[0])(i,j,k,m)*numerical_pde->DPhi(m,xi,d);
      derivative_rhou_q +=((*uw)[q])(i,j,k,m)*numerical_pde->DPhi(m,xi,d);
      rho+=((*uw)[0])(i,j,k,m)*numerical_pde->Phi(m,xi);
      rhou_q+=((*uw)[q])(i,j,k,m)*numerical_pde->Phi(m,xi);
    }
    else
    {
      derivative+=((*uw)[0])(i,j,k,m)*numerical_pde->DPhi(m,xi,d);
    }
  }
  
  if(q!=0){
    derivative =(rho*derivative_rhou_q-rhou_q*derivative_rho)/(std::pow(rho,2));
  }
  
  return derivative;
}

amrex::Real Compressible_Euler::get_D2U_from_U_w(int d1, int d2, int q, int i, int j, int k,
                                                amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                                amrex::Vector<amrex::Real> xi) const
{
  //get the derivative of the specified solution component along specified direction
  //used for Euler equations, divide by density ((*uw)[0])(i,j,k,m)) to obtain just velocity
  //in case we need derivative of density, then we do not divide by it
  //d^2/dxi dxj -> dxi==dx1, dxj==dx2
  amrex::Real derivative = 0.0;
  amrex::Real derivative1_rho = 0.0;
  amrex::Real derivative1_rhou_q = 0.0;
  amrex::Real derivative2_rho = 0.0;
  amrex::Real derivative2_rhou_q = 0.0;
  amrex::Real d_derivative_rhou_q = 0.0;
  amrex::Real d_derivative_rho = 0.0;
  amrex::Real rho=0.0;
  amrex::Real rhou_q = 0.0;
  for(int m=0; m<numerical_pde->Np; ++m)
  { 
    if(q!=0){
      d_derivative_rho+=((*uw)[0])(i,j,k,m)*numerical_pde->DDPhi(m,xi,d1,d2);
      d_derivative_rhou_q +=((*uw)[q])(i,j,k,m)*numerical_pde->DDPhi(m,xi,d1,d2);
      derivative1_rho+=((*uw)[0])(i,j,k,m)*numerical_pde->DPhi(m,xi,d1);
      derivative1_rhou_q +=((*uw)[q])(i,j,k,m)*numerical_pde->DPhi(m,xi,d1);
      derivative2_rho+=((*uw)[0])(i,j,k,m)*numerical_pde->DPhi(m,xi,d2);
      derivative2_rhou_q +=((*uw)[q])(i,j,k,m)*numerical_pde->DPhi(m,xi,d2);
      rho+=((*uw)[0])(i,j,k,m)*numerical_pde->Phi(m,xi);
      rhou_q+=((*uw)[q])(i,j,k,m)*numerical_pde->Phi(m,xi);
    }
    else
    {
      derivative+=((*uw)[0])(i,j,k,m)*numerical_pde->DDPhi(m,xi,d1,d2);
    }
  }
  
  if(q!=0){
    derivative =(
    std::pow(rho,2)*(rho*d_derivative_rhou_q+derivative2_rho*derivative1_rhou_q
                    -rhou_q*d_derivative_rho-derivative2_rhou_q*derivative1_rho)
                    -2*rho*derivative2_rho*(rho*derivative1_rhou_q-
                    rhou_q*derivative1_rho))/(std::pow(rho,4));
  }
  
  return derivative;
}

amrex::Real Compressible_Euler::smooth_discontinuity(amrex::Real xi, amrex::Real a, 
                                                     amrex::Real b, amrex::Real s) const
{
  amrex::Real d; 
  d=a+0.5*(1.0+std::tanh(s*xi))*(b-a);
  
  return d;
}

amrex::Real Compressible_Euler::Pressure(amrex::Vector<amrex::Array4<const amrex::Real>>* u, 
                                        int i, int j, int k,int m) const
{
  amrex::Real prs =0.0;
  for(int d=0; d<AMREX_SPACEDIM; ++d)//velocity norm
  {
    prs+=(std::pow(((*u)[d+1])(i,j,k,m),2.0)/((*u)[0])(i,j,k,m));
  }
  prs*=(-0.5);
  prs+=((*u)[3])(i,j,k,m);
  prs*=(gamma_adiab-1.0);
  return prs;
}

amrex::Real Compressible_Euler::Pressure(amrex::Vector<amrex::Array4< amrex::Real>>* u, 
                                        int i, int j, int k,int m) const
{
  amrex::Real prs =0.0;
  for(int d=0; d<AMREX_SPACEDIM; ++d)//velocity norm
  {
    prs+=(std::pow(((*u)[d+1])(i,j,k,m),2.0)/((*u)[0])(i,j,k,m));
  }
  prs*=(-0.5);
  prs+=((*u)[3])(i,j,k,m);
  prs*=(gamma_adiab-1.0);
  return prs;
}

amrex::Real Compressible_Euler::Soundspeed(amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                          int i, int j, int k, int m) const
{
  //return the pointwise value of soundspeed, i.e at quadrature/interpolation point
  amrex::Real c;
  amrex::Real prs=Pressure(u,i,j,k,m);
  c = std::sqrt(gamma_adiab*(prs/((*u)[0])(i,j,k,m)));
  return c;
}

amrex::Real Compressible_Euler::Soundspeed(amrex::Vector<amrex::Array4< amrex::Real>>* u,
                                          int i, int j, int k, int m) const
{
  //return the pointwise value of soundspeed, i.e at quadrature/interpolation point
  amrex::Real c;
  amrex::Real prs=Pressure(u,i,j,k,m);
  
  c = std::sqrt(gamma_adiab*(prs/((*u)[0])(i,j,k,m)));
  return c;
}

amrex::Real Compressible_Euler::pde_CFL(int d,int m,int i, int j, int k,
                                        amrex::Vector<amrex::Array4<const amrex::Real>>* u) const
{
  amrex::Real lambda;
  amrex::Real c=Soundspeed(u,i, j, k, 0);
  if(d==0)
  {lambda = std::abs(((*u)[1])(i,j,k,0)/((*u)[0])(i,j,k,0))+c;}
  else if(d==1)
  {lambda = std::abs(((*u)[2])(i,j,k,0)/((*u)[0])(i,j,k,0))+c;}
  else if(d==2)
  {lambda = std::abs(((*u)[3])(i,j,k,0)/((*u)[0])(i,j,k,0))+c;}
  return lambda;
} 

void Compressible_Euler::pde_BC(int lev, int dim,int side, int q,  int quad_pt_idx,
                                const IntVect& iv, const int dcomp, const int ncomp,
                                amrex::Vector<amrex::Real>* Ubc, 
                                amrex::Vector<amrex::Real>* Ubc_valid) const 
                               
{                                                 
  //when applying BC to EUler components, we specify e.g BC for u1, but we evolve 
  //rho_u1, therefore we need to map that BC to the 
  //solution component
  //lines below allow to recover the value of the other solution components
  //depending on the model used we might want to use more elaborate ways/set of points.
  //Below a general formulation
  //Ubc : primitive bc 
  //Ubc_valid: conserved bc
  
  //Ubc is equivalent to modes of closest valid cell evalauted at interface with ghost cell
  amrex::Vector<const amrex::MultiFab *> state_u_w(numerical_pde->Q); 
  amrex::Vector<const amrex::FArrayBox *> fab_u_w(numerical_pde->Q);
  amrex::Vector<amrex::Array4<const amrex::Real>> uw(numerical_pde->Q);   
  
  for(int qq=0; qq<numerical_pde->Q; ++qq){
    //cannot use q because is a bc class variable
    state_u_w[qq]=&(numerical_pde->U_w[lev][qq]);
  } 

  amrex::Vector<amrex::Real> xi_ref_bd(AMREX_SPACEDIM);
  amrex::Vector<amrex::Real> xi_ref_valid(AMREX_SPACEDIM);
  for (int d = 0; d < AMREX_SPACEDIM; ++d){
    xi_ref_bd[d]=numerical_pde->xi_ref_GLquad_L2proj[quad_pt_idx][d];
    xi_ref_valid[d]=numerical_pde->xi_ref_GLquad_L2proj[quad_pt_idx][d];
  }
  
  if(side==-1){xi_ref_bd[dim]=-1.0;}
  else if(side==1){xi_ref_bd[dim]=1.0;}
  
  int flag = amrex::MFIter::allowMultipleMFIters(true); 
  for (MFIter mfi(numerical_pde->U_w[lev][q]); mfi.isValid(); ++mfi)
  {
    Box const& bx = mfi.growntilebox();
    for(int qq=0 ; qq<numerical_pde->Q; ++qq){      
      fab_u_w[qq] = state_u_w[qq]->fabPtr(mfi);
      uw[qq] = fab_u_w[qq]->const_array();
    } 

    if(side==-1 && bx.contains(iv+IntVect::TheDimensionVector(dim))){
      for(int qq=0 ; qq<numerical_pde->Q; ++qq){ 
        amrex::Real sum=0.0;
        amrex::Real sum_valid=0.0;
        for (int n = 0; n < ncomp; ++n){
          //evaluation at itnerface with boundary
          sum+=(uw[qq])(iv+IntVect::TheDimensionVector(dim),n+dcomp)
              *(numerical_pde->Phi(n+dcomp, xi_ref_bd));
          //evaluation at cell center
          sum_valid+=(uw[qq])(iv+IntVect::TheDimensionVector(dim),n+dcomp)
                    *(numerical_pde->Phi(n+dcomp, xi_ref_valid));
        }  
        (*Ubc)[qq]=sum;
        (*Ubc_valid)[qq]=sum_valid;      
      }
      break;
    }
    else if(side==1 && bx.contains(iv-IntVect::TheDimensionVector(dim))){     
      for(int qq=0 ; qq<numerical_pde->Q; ++qq){ 
        amrex::Real sum=0.0;
        amrex::Real sum_valid=0.0;
        for (int n = 0; n < ncomp; ++n){
          //evaluation at itnerface with boundary
          sum+=(uw[qq])(iv-IntVect::TheDimensionVector(dim),n+dcomp)
              *(numerical_pde->Phi(n+dcomp, xi_ref_bd));
          //evaluation at cell center
          sum_valid+=(uw[qq])(iv-IntVect::TheDimensionVector(dim),n+dcomp)
                    *(numerical_pde->Phi(n+dcomp, xi_ref_valid));
        }  
        (*Ubc)[qq]=sum;
        (*Ubc_valid)[qq]=sum_valid;  
      }
      break;
    }
  }
  flag = amrex::MFIter::allowMultipleMFIters(false);  
}

amrex::Real Compressible_Euler::pde_BC_gDirichlet(int q, int dim, const IntVect& iv, int quad_pt_idx, 
                                                  const int dcomp,const int ncomp, Array4<Real> const& dest, 
                                                  GeometryData const& geom, int side, int lev) const
{

  //NB: currently Dirichlet BC is not evaluated differently depending on interpolation point location 
  //(intepr point xi_ref_GLquad_L2proj[quad_pt_idx])
  //but this can be usefull in case
  //we want to (application dependant) define functions that are evalauted at quadrature point (e.g boundary radiation stuff)
  //in that scenario the successive L2 projection make sense, atm its a bit overkill
  //also here angular momentum we jsut shift location towards interface
  //in short, in case pde_BC is used, then projection will make sense 
  
  const auto lo = geom.Domain().smallEnd();
  const auto hi = geom.Domain().bigEnd(); 
  const auto dx = numerical_pde->Geom(lev).CellSizeArray();  
  
  amrex::Real bc_val, bc;
  
  amrex::Real delta= dx[dim];
  
#if(AMREX_SPACEDIM ==2)

  amrex::Real gD_rho, gD_u1, gD_u2, gD_e;
  //Recover the primitive gradients  
  if(side==-1){
    gD_rho =numerical_pde->gDbc_lo[0][dim];
    gD_u1 =numerical_pde->gDbc_lo[1][dim];
    gD_u2 =numerical_pde->gDbc_lo[2][dim];
    gD_e =numerical_pde->gDbc_lo[3][dim];
  }
  else if(side==1){
    gD_rho =numerical_pde->gDbc_hi[0][dim];
    gD_u1 =numerical_pde->gDbc_hi[1][dim];
    gD_u2 =numerical_pde->gDbc_hi[2][dim];
    gD_e =numerical_pde->gDbc_hi[3][dim];  
  }
  
  //Compute the conserved values
  if(q==0){bc=gD_rho;}
  else if(q==1){bc=gD_rho*gD_u1;}
  else if(q ==2){bc=gD_rho*gD_u2;}
  else if(q==3){
    bc = (gD_e/(gamma_adiab-1.0))+0.5*(gD_rho*std::pow(gD_u1,2.0)
          +gD_rho*std::pow(gD_u2,2.0));
  }
  else if(q==4){
    amrex::Real x1_bc = lo[0] + (iv[0]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[0]+0.5)*geom.CellSize()[0]
                      +(amrex::Real)side*geom.CellSize()[0]*0.5;
                      
    amrex::Real x2_bc = lo[1] + (iv[1]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[1]+0.5)*geom.CellSize()[1]
                      +(amrex::Real)side*geom.CellSize()[1]*0.5;
                      
    bc =x1_bc*gD_u2-x2_bc*gD_u1;
  }

  bc_val = bc;
  
#elif(AMREX_SPACEDIM ==3)
  amrex::Real gD_rho, gD_u1, gD_u2,gD_u3, gD_e;
  //Recover the primitive gradients  
  if(side==-1){
    gD_rho =numerical_pde->gDbc_lo[0][dim];
    gD_u1 =numerical_pde->gDbc_lo[1][dim];
    gD_u2 =numerical_pde->gDbc_lo[2][dim];
    gD_u3 =numerical_pde->gDbc_lo[3][dim];
    gD_e =numerical_pde->gDbc_lo[4][dim];
  }
  else if(side==1){
    gD_rho =numerical_pde->gDbc_hi[0][dim];
    gD_u1 =numerical_pde->gDbc_hi[1][dim];
    gD_u2 =numerical_pde->gDbc_hi[2][dim];
    gD_u3 =numerical_pde->gDbc_hi[3][dim];
    gD_e =numerical_pde->gDbc_hi[4][dim];  
  }
  
  //Compute the conserved values
  if(q==0){bc=gD_rho;}
  else if(q==1){bc=gD_rho*gD_u1;}
  else if(q ==2){bc=gD_rho*gD_u2;}
  else if(q ==2){bc=gD_rho*gD_u3;}
  else if(q==4){
    bc = (gD_e/(gamma_adiab-1.0))+0.5*(gD_rho*std::pow(gD_u1,2.0)+
          gD_rho*std::pow(gD_u2,2.0)+gD_rho*std::pow(gD_u3,2.0));
  }
  else if(q==5){
    amrex::Real x3_bc = lo[2] + (iv[2]-(amrex::Real)side
                        *(IntVect::TheDimensionVector(dim))[2]+0.5) * geom.CellSize()[2]
                        +(amrex::Real)side*geom.CellSize()[2]*0.5;
                      
    amrex::Real x2_bc = lo[1] + (iv[1]-(amrex::Real)side
                        *(IntVect::TheDimensionVector(dim))[1]+0.5)*geom.CellSize()[1]
                        +(amrex::Real)side*geom.CellSize()[1]*0.5;
                      
    bc =x2_bc*gD_u3-x3_bc*gD_u2;
  }
  else if(q==7){
    amrex::Real x1_bc = lo[0] + (iv[0]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[0]+0.5)*geom.CellSize()[0]
                      +(amrex::Real)side*geom.CellSize()[0]*0.5;
                      
    amrex::Real x3_bc = lo[2] + (iv[2]-(amrex::Real)side
                        *(IntVect::TheDimensionVector(dim))[2]+0.5) * geom.CellSize()[2]
                        +(amrex::Real)side*geom.CellSize()[2]*0.5;
                      
    bc =x3_bc*gD_u1-x1_bc*gD_u3;
  }
  else if(q==7){
    amrex::Real x1_bc = lo[0] + (iv[0]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[0]+0.5)*geom.CellSize()[0]
                      +(amrex::Real)side*geom.CellSize()[0]*0.5;
                      
    amrex::Real x2_bc = lo[1] + (iv[1]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[1]+0.5)*geom.CellSize()[1]
                      +(amrex::Real)side*geom.CellSize()[1]*0.5;
                      
    bc =x1_bc*gD_u2-x2_bc*gD_u1;
  }
  
  bc_val = bc;
  
#endif

  return bc_val;
}
 
amrex::Real Compressible_Euler::pde_BC_gNeumann(int q, int dim,const IntVect& iv, 
                                                int quad_pt_idx, const int dcomp,const int ncomp,
                                                Array4<Real> const& dest,GeometryData const& geom, 
                                                int side, int lev) const
{
  //Know current q im looping over, the side and dimension
  //use amrdg pointer to access the g_N value sspecified for all componetns
  //if side low use gbc_lo, if side hi use gbc_hi
  //construct all gradients independently of what q we are solving for
  //For momentum we need to know the modes evaluated at boundary location, use itnerface
  //using modes of inner valid cell (use pde_BC)
  //then , depending on q we just take one of the computed conserved gradient that we need
  //and use that as BC function to compute inner outer vecctor and finally BCs  

  const auto lo = geom.Domain().smallEnd();
  const auto hi = geom.Domain().bigEnd(); 
  const auto dx = numerical_pde->Geom(lev).CellSizeArray();
  
  amrex::Real bc_val, grad;
  
  amrex::Real delta= dx[dim];

#if(AMREX_SPACEDIM ==2)
  amrex::Real gN_rho, gN_u1, gN_u2, gN_e;
  //Recover the primitive gradients  
  if(side==-1){
    gN_rho =numerical_pde->gNbc_lo[0][dim];
    gN_u1 =numerical_pde->gNbc_lo[1][dim];
    gN_u2 =numerical_pde->gNbc_lo[2][dim];
    gN_e =numerical_pde->gNbc_lo[3][dim];
  }
  else if(side==1){
    gN_rho =numerical_pde->gNbc_hi[0][dim];
    gN_u1 =numerical_pde->gNbc_hi[1][dim];
    gN_u2 =numerical_pde->gNbc_hi[2][dim];
    gN_e =numerical_pde->gNbc_hi[3][dim];  
  }
  
  //Recover the value at the boundary interface from the closes 
  //inner valid cell. Also polynomial at center of cell valid is computed
  amrex::Vector<amrex::Real> Ubc(numerical_pde->Q,0.0);
  amrex::Vector<amrex::Real> Ubc_valid(numerical_pde->Q,0.0);
  pde_BC(lev,dim,side,q,quad_pt_idx, iv, dcomp, ncomp, &Ubc,&Ubc_valid);
  
  //Compute the conserved gradients
  if(q==0){
    grad=gN_rho;
  }
  else if(q==1){ 
    grad = Ubc[0]*gN_u1+(Ubc[q]/Ubc[0])*gN_rho;
  }
  else if(q==2){ 
    grad = Ubc[0]*gN_u2+(Ubc[q]/Ubc[0])*gN_rho;
  }
  else if(q==3){
    //NB: gN_e is the pressure gradient!
    grad = (gN_e/(gamma_adiab-1.0))+0.5*(std::pow((Ubc[1]/Ubc[0]),2.0)*gN_rho
           +std::pow((Ubc[2]/Ubc[0]),2.0)*gN_rho)+(Ubc[1]*gN_u1+Ubc[2]*gN_u2);

  }
  else if(q==4)
  {
    amrex::Real x1c = lo[0] + (iv[0]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[0]+0.5) * dx[0];
                                           
    amrex::Real x2c = lo[1] + (iv[1]-(amrex::Real)side
                        *(IntVect::TheDimensionVector(dim))[1]+0.5) * dx[1];
     
    //need to use same location (i.e interface) where the other gradients have been defined
    amrex::Vector<amrex::Real> xi_tmp(AMREX_SPACEDIM);
    for (int d = 0; d < AMREX_SPACEDIM; ++d){
      xi_tmp[d]=numerical_pde->xi_ref_GLquad_L2proj[quad_pt_idx][d];
    }    
    if(side==-1){xi_tmp[dim]=-1.0;}
    else if(side==1){xi_tmp[dim]=1.0;}
  
    amrex::Real x1_bc = 0.5*dx[0]*(xi_tmp[0])+x1c;
    amrex::Real x2_bc = 0.5*dx[1]*(xi_tmp[1])+x2c;
     
    //what abt shift position towards boundary ("i.e cell centered at boundary")by half a cell?
      
    amrex::Real grad_rho_u1 = Ubc[0]*gN_u1+(Ubc[1]/Ubc[0])*gN_rho;
    amrex::Real grad_rho_u2 = Ubc[0]*gN_u2+(Ubc[2]/Ubc[0])*gN_rho;

    amrex::Real dx_nk_dx_j,dx_lk_dx_j;

    if(dim=0){dx_nk_dx_j =1.0; dx_lk_dx_j=0.0;}
    else if(dim==1){dx_nk_dx_j =0.0; dx_lk_dx_j=1.0;}

    grad = Ubc[2]*(dx_nk_dx_j)+x1_bc*(grad_rho_u2)-Ubc[1]*( dx_lk_dx_j)-x2_bc*(grad_rho_u1);
  }

  bc_val = Ubc_valid[q] + grad * delta;

#elif(AMREX_SPACEDIM ==3)
  amrex::Real gN_rho, gN_u1, gN_u2,gN_u3, gN_e;
  //Recover the primitive gradients  
  if(side==-1){
    gN_rho =numerical_pde->gNbc_lo[0][dim];
    gN_u1 =numerical_pde->gNbc_lo[1][dim];
    gN_u2 =numerical_pde->gNbc_lo[2][dim];
    gN_u3 =numerical_pde->gNbc_lo[3][dim];
    gN_e =numerical_pde->gNbc_lo[4][dim];
  }
  else if(side==1){
    gN_rho =numerical_pde->gNbc_hi[0][dim];
    gN_u1 =numerical_pde->gNbc_hi[1][dim];
    gN_u2 =numerical_pde->gNbc_hi[2][dim];
    gN_u3 =numerical_pde->gNbc_hi[3][dim];    
    gN_e =numerical_pde->gNbc_hi[4][dim];  
  }
  
  //Recover the value at the boundary interface from the closes 
  //inner valid cell. Also polynomial at center of cell valid is computed
  amrex::Vector<amrex::Real> Ubc(numerical_pde->Q,0.0);
  amrex::Vector<amrex::Real> Ubc_valid(numerical_pde->Q,0.0);
  pde_BC(lev,dim,side,q,quad_pt_idx, iv, dcomp, ncomp, &Ubc,&Ubc_valid);
    
  //Compute the conserved gradients
  if(q==0){
    grad=gN_rho;
  }
  else if(q==1){ 
    grad = Ubc[0]*gN_u1+(Ubc[q]/Ubc[0])*gN_rho;
  }
  else if(q==2){ 
    grad = Ubc[0]*gN_u2+(Ubc[q]/Ubc[0])*gN_rho;
  }
  else if(q==3){ 
    grad = Ubc[0]*gN_u3+(Ubc[q]/Ubc[0])*gN_rho;
  }
  else if(q==4){
    //NB: gN_e is the pressure gradient!    
    grad = (gN_e/(gamma_adiab-1.0))+0.5*(std::pow((Ubc[1]/Ubc[0]),2.0)*gN_rho
            +std::pow((Ubc[2]/Ubc[0]),2.0)*gN_rho+std::pow((Ubc[3]/Ubc[0]),2.0)*gN_rho) 
            +(Ubc[1]*gN_u1+Ubc[2]*gN_u2+Ubc[3]*gN_u3);      
  }
  else if(q==5)
  {
    amrex::Real x3c = lo[2] + (iv[2]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[2]+0.5) * dx[2];
                        
                        
    amrex::Real x2c = lo[1] + (iv[1]-(amrex::Real)side
                        *(IntVect::TheDimensionVector(dim))[1]+0.5) * dx[1];

    amrex::Vector<amrex::Real> xi_tmp(AMREX_SPACEDIM);
    for (int d = 0; d < AMREX_SPACEDIM; ++d){
      xi_tmp[d]=numerical_pde->xi_ref_GLquad_L2proj[quad_pt_idx][d];
    }    
    if(side==-1){xi_tmp[dim]=-1.0;}
    else if(side==1){xi_tmp[dim]=1.0;}
    
    amrex::Real x3_bc=0.5*dx[2]*(xi_tmp[2])+x3c;
    amrex::Real x2_bc=0.5*dx[1]*(xi_tmp[1])+x2c;
    
    amrex::Real grad_rho_u2 = Ubc[0]*gN_u2+(Ubc[2]/Ubc[0])*gN_rho;
    amrex::Real grad_rho_u3 = Ubc[0]*gN_u3+(Ubc[3]/Ubc[0])*gN_rho;
    
    amrex::Real dx_nk_dx_j,dx_lk_dx_j;
    
    if(dim=0){dx_nk_dx_j =0.0; dx_lk_dx_j=0.0;}
    else if(dim==1){dx_nk_dx_j =1.0; dx_lk_dx_j=0.0;}
    else if(dim==2){dx_nk_dx_j =0.0; dx_lk_dx_j=1.0;}

    grad = Ubc[3]*(dx_nk_dx_j)+x2_bc*(grad_rho_u3)-Ubc[2]*( dx_lk_dx_j)-x3_bc*(grad_rho_u2);

  }
  else if(q==6)
  {
    amrex::Real x3c = lo[2] + (iv[2]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[2]+0.5) * dx[2];
                        
    amrex::Real x1c = lo[0] + (iv[0]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[0]+0.5) * dx[0];

    amrex::Vector<amrex::Real> xi_tmp(AMREX_SPACEDIM);
    for (int d = 0; d < AMREX_SPACEDIM; ++d){
      xi_tmp[d]=numerical_pde->xi_ref_GLquad_L2proj[quad_pt_idx][d];
    }    
    if(side==-1){xi_tmp[dim]=-1.0;}
    else if(side==1){xi_tmp[dim]=1.0;}

    amrex::Real x3_bc=0.5*dx[2]*(xi_tmp[2])+x3c;
    amrex::Real x1_bc = 0.5*dx[0]*(xi_tmp[0])+x1c;
    
    amrex::Real grad_rho_u1 = Ubc[0]*gN_u1+(Ubc[1]/Ubc[0])*gN_rho;
    amrex::Real grad_rho_u3 = Ubc[0]*gN_u3+(Ubc[3]/Ubc[0])*gN_rho;
    
    amrex::Real dx_nk_dx_j,dx_lk_dx_j;
    
    if(dim=0){dx_nk_dx_j =0.0; dx_lk_dx_j=1.0;}
    else if(dim==1){dx_nk_dx_j =0.0; dx_lk_dx_j=0.0;}
    else if(dim==2){dx_nk_dx_j =1.0; dx_lk_dx_j=0.0;}
   
    grad = Ubc[1]*(dx_nk_dx_j)+x3_bc*(grad_rho_u1)-Ubc[3]*( dx_lk_dx_j)-x1_bc*(grad_rho_u3);
  }
  else if(q==7)
  {
    amrex::Real x1c = lo[0] + (iv[0]-(amrex::Real)side
                      *(IntVect::TheDimensionVector(dim))[0]+0.5) * dx[0];
                                           
    amrex::Real x2c = lo[1] + (iv[1]-(amrex::Real)side
                        *(IntVect::TheDimensionVector(dim))[1]+0.5) * dx[1];
     
    amrex::Vector<amrex::Real> xi_tmp(AMREX_SPACEDIM);
    for (int d = 0; d < AMREX_SPACEDIM; ++d){
      xi_tmp[d]=numerical_pde->xi_ref_GLquad_L2proj[quad_pt_idx][d];
    }    
    if(side==-1){xi_tmp[dim]=-1.0;}
    else if(side==1){xi_tmp[dim]=1.0;}
  
    amrex::Real x1_bc = 0.5*dx[0]*(xi_tmp[0])+x1c;
    amrex::Real x2_bc = 0.5*dx[1]*(xi_tmp[1])+x2c;
                      
    amrex::Real grad_rho_u1 = Ubc[0]*gN_u1+(Ubc[1]/Ubc[0])*gN_rho;
    amrex::Real grad_rho_u2 = Ubc[0]*gN_u2+(Ubc[2]/Ubc[0])*gN_rho;
    
    amrex::Real dx_nk_dx_j,dx_lk_dx_j;
    
    if(dim=0){dx_nk_dx_j =1.0; dx_lk_dx_j=0.0;}
    else if(dim==1){dx_nk_dx_j =0.0; dx_lk_dx_j=1.0;}
    else if(dim==2){dx_nk_dx_j =0.0; dx_lk_dx_j=0.0;}
    
    grad = Ubc[2]*(dx_nk_dx_j)+x1_bc*(grad_rho_u2)-Ubc[1]*( dx_lk_dx_j)-x2_bc*(grad_rho_u1);
  }
  
  bc_val = Ubc_valid[q] + grad * delta;

#endif                        
  return bc_val;
}

amrex::Vector<amrex::Vector<amrex::Real>> 
Compressible_Euler::pde_EV_Rmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4<const amrex::Real>>* u) const
{
  amrex::Real c = Soundspeed(u, i,j,k,m);
  amrex::Vector<amrex::Vector<amrex::Real>> R_EV;
  R_EV.resize(Q_model_unique,amrex::Vector<amrex::Real>(Q_model_unique));
  
  #if (AMREX_SPACEDIM == 2)
  amrex::Real u1 =((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real u2 =((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real ek  =0.5*(std::pow(u1,2)+std::pow(u2,2));
  amrex::Real h  =(std::pow(c,2)/(gamma_adiab-1.0))+ek;
  
  if(d==0){
    R_EV =  {
              {1.0,1.0,1.0,0.0},
              {u1-c,u1,u1+c,0.0},
              {u2,u2,u2,-1.0},
              {h-c*u1,ek,h+c*u1,-u2}
            };
  }
  else if(d==1){
    R_EV =  {
              {1.0,1.0,1.0,0.0},
              {u1,u1,u1,1.0},
              {u2-c,u2,u2+c,0.0},
              {h-c*u2,ek,h+c*u2,u1}
            };
  }
  #elif (AMREX_SPACEDIM == 3)
  amrex::Real u1 =((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real u2 =((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real u3 =((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real ek  =0.5*(std::pow(u1,2)+std::pow(u2,2)+std::pow(u3,2));
  amrex::Real h  =(std::pow(c,2)/(gamma_adiab-1.0))+ek;
  
  if(d==0){
    R_EV =  {
              {1.0,1.0,1.0,0.0,0.0},
              {u1-c,u1,u1+c,0.0,0.0},
              {u2,u2,u2,-1.0,0.0},
              {u3,u3,u3,0.0,1.0},
              {h-c*u1,ek,h+c*u1,-u2,u3}
            }
  };
  else if(d==1){
    R_EV =  {
              {1.0,1.0,1.0,0.0,0.0},
              {u1,u1,u1,1.0,0.0},
              {u2-c,u2,u2+c,0.0,0.0},
              {u3,u3,u3,0.0,-1.0},
              {h-c*u2,ek,h+c*u2,u1,-u3}
            };
  }
  else if(d==2){
    R_EV =  {
              {1.0,1.0,1.0,0.0,0.0},
              {u1,u1,u1,-1.0,0.0},
              {u2,u2,u2,0.0,1.0},
              {u3-c,u3,u3+c,0.0,0.0},
              {h-c*u3,ek,h+c*u3,-u1,u2}
            };
  }
  #endif
  
  return R_EV;  
}

amrex::Vector<amrex::Vector<amrex::Real>> 
Compressible_Euler::pde_EV_Rmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4<amrex::Real>>* u) const
{
  amrex::Real c = Soundspeed(u, i,j,k,m);
  amrex::Vector<amrex::Vector<amrex::Real>> R_EV;
  R_EV.resize(Q_model_unique,amrex::Vector<amrex::Real>(Q_model_unique));
  
  #if (AMREX_SPACEDIM == 2)
  amrex::Real u1 =((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real u2 =((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real ek  =0.5*(std::pow(u1,2)+std::pow(u2,2));
  amrex::Real h  =(std::pow(c,2)/(gamma_adiab-1.0))+ek;
  
  if(d==0){
    R_EV =  {
              {1.0,1.0,1.0,0.0},
              {u1-c,u1,u1+c,0.0},
              {u2,u2,u2,-1.0},
              {h-c*u1,ek,h+c*u1,-u2}
            };
  }
  else if(d==1){
    R_EV =  {
              {1.0,1.0,1.0,0.0},
              {u1,u1,u1,1.0},
              {u2-c,u2,u2+c,0.0},
              {h-c*u2,ek,h+c*u2,u1}
            };
  }
  #elif (AMREX_SPACEDIM == 3)
  amrex::Real u1 =((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real u2 =((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real u3 =((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
  amrex::Real ek  =0.5*(std::pow(u1,2)+std::pow(u2,2)+std::pow(u3,2));
  amrex::Real h  =(std::pow(c,2)/(gamma_adiab-1.0))+ek;
  
  if(d==0){
    R_EV =  {
              {1.0,1.0,1.0,0.0,0.0},
              {u1-c,u1,u1+c,0.0,0.0},
              {u2,u2,u2,-1.0,0.0},
              {u3,u3,u3,0.0,1.0},
              {h-c*u1,ek,h+c*u1,-u2,u3}
            };
  }
  else if(d==1){
    R_EV =  {
              {1.0,1.0,1.0,0.0,0.0},
              {u1,u1,u1,1.0,0.0},
              {u2-c,u2,u2+c,0.0,0.0},
              {u3,u3,u3,0.0,-1.0},
              {h-c*u2,ek,h+c*u2,u1,-u3}
            };
  }
  else if(d==2){
    R_EV =  {
              {1.0,1.0,1.0,0.0,0.0},
              {u1,u1,u1,-1.0,0.0},
              {u2,u2,u2,0.0,1.0},
              {u3-c,u3,u3+c,0.0,0.0},
              {h-c*u3,ek,h+c*u3,-u1,u2}
            };
  }
  #endif
  
  return R_EV;  
}

amrex::Vector<amrex::Vector<amrex::Real>> 
Compressible_Euler::pde_EV_Lmatrix(int d,int m, int i, int j, int k, 
                                  amrex::Vector<amrex::Array4<const amrex::Real>>* u) const
{
  amrex::Real c = Soundspeed(u, i,j,k,m);
  amrex::Vector<amrex::Vector<amrex::Real>> L_EV;
  L_EV.resize(Q_model_unique,amrex::Vector<amrex::Real>(Q_model_unique));
  #if (AMREX_SPACEDIM == 2)
    amrex::Real u1 =((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real u2 =((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real ek  =0.5*(std::pow(u1,2)+std::pow(u2,2));
    amrex::Real h  =(std::pow(c,2)/(gamma_adiab-1.0))+ek;
    amrex::Real beta = 1.0/(2.0*std::pow(c,2));
    amrex::Real gamma = gamma_adiab-1.0;
    amrex::Real phi = gamma*ek;
  
    if(d==0){
      L_EV =  {
                {beta*(phi+c*u1),-beta*(gamma*u1+c),-beta*gamma*u2,beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,-2.0*beta*gamma},
                {beta*(phi-c*u1),-beta*(gamma*u1-c),-beta*(gamma*u2),beta*gamma},
                {u2,0.0,-1.0,0.0}
              };
    }
    else if(d==1){
      L_EV =  {
                {beta*(phi+c*u2),-beta*gamma*u1,-beta*(gamma*u2+c),beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,-2.0*beta*gamma},
                {beta*(phi-c*u2),-beta*(gamma*u1),-beta*(gamma*u2-c),beta*gamma},
                {-u1,1.0,0.0,0.0}
              };
    }
  #elif (AMREX_SPACEDIM == 3)
    amrex::Real u1 =((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real u2 =((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real u3 =((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real ek  =0.5*(std::pow(u1,2)+std::pow(u2,2)+std::pow(u3,2));
    amrex::Real h  =(std::pow(c,2)/(gamma_adiab-1.0))+ek;
    amrex::Real beta = 1.0/(2.0*std::pow(c,2));
    amrex::Real gamma = gamma_adiab-1.0;
    amrex::Real phi = gamma*ek;
  
    if(d==0){
      L_EV =  {
                {beta*(phi+c*u1),-beta*(gamma*u1+c),-beta*gamma*u2,-beta*gamma*u3,beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,2.0*beta*gamma*u3,-2.0*beta*gamma},
                {beta*(phi-c*u1),-beta*(gamma*u1-c),-beta*(gamma*u2),-beta*gamma*u3,beta*gamma},
                {u2,0.0,-1.0,0.0,0.0},
                {-u3,0.0,0.0,1.0,0.0}
              };
    }
    else if(d==1){
      L_EV =  {
                {beta*(phi+c*u2),-beta*gamma*u1,-beta*(gamma*u2+c),-beta*gamma*u3,beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,2.0*beta*gamma*u3,-2.0*beta*gamma},
                {beta*(phi-c*u2),-beta*(gamma*u1),-beta*(gamma*u2-c),-beta*gamma*u3,beta*gamma},
                {-u1,1.0,0.0,0.0,0.0},
                {u3,0.0,0.0,-1.0,0.0}
              };
    }
    else if(d==2){
      L_EV =  {
                {beta*(phi+c*u3),-beta*gamma*u1,-beta*gamma*u2,-beta*(gamma*u3+c),beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,2.0*beta*gamma*u3,-2.0*beta*gamma},
                {beta*(phi-c*u3),-beta*(gamma*u1),-beta*gamma*u2,-beta*(gamma*u3-c),beta*gamma},
                {u1,-1.0,0.0,0.0,0.0},
                {-u2,0.0,1.0,0.0,0.0}
              };
    }
  #endif 

  return L_EV;
}

amrex::Vector<amrex::Vector<amrex::Real>> 
Compressible_Euler::pde_EV_Lmatrix(int d,int m, int i, int j, int k, 
                                  amrex::Vector<amrex::Array4<amrex::Real>>* u) const
{
  amrex::Real c = Soundspeed(u, i,j,k,m);
  amrex::Vector<amrex::Vector<amrex::Real>> L_EV;
  L_EV.resize(Q_model_unique,amrex::Vector<amrex::Real>(Q_model_unique));
  #if (AMREX_SPACEDIM == 2)
    amrex::Real u1 =((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real u2 =((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real ek  =0.5*(std::pow(u1,2)+std::pow(u2,2));
    amrex::Real h  =(std::pow(c,2)/(gamma_adiab-1.0))+ek;
    amrex::Real beta = 1.0/(2.0*std::pow(c,2));
    amrex::Real gamma = gamma_adiab-1.0;
    amrex::Real phi = gamma*ek;
  
    if(d==0){
      L_EV =  {
                {beta*(phi+c*u1),-beta*(gamma*u1+c),-beta*gamma*u2,beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,-2.0*beta*gamma},
                {beta*(phi-c*u1),-beta*(gamma*u1-c),-beta*(gamma*u2),beta*gamma},
                {u2,0.0,-1.0,0.0}
              };
    }    
    else if(d==1){
      L_EV =  {
                {beta*(phi+c*u2),-beta*gamma*u1,-beta*(gamma*u2+c),beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,-2.0*beta*gamma},
                {beta*(phi-c*u2),-beta*(gamma*u1),-beta*(gamma*u2-c),beta*gamma},
                {-u1,1.0,0.0,0.0}
              };
    }
  #elif (AMREX_SPACEDIM == 3)
    amrex::Real u1 =((*u)[1])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real u2 =((*u)[2])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real u3 =((*u)[3])(i,j,k,m)/((*u)[0])(i,j,k,m) ;
    amrex::Real ek  =0.5*(std::pow(u1,2)+std::pow(u2,2)+std::pow(u3,2));
    amrex::Real h  =(std::pow(c,2)/(gamma_adiab-1.0))+ek;
    amrex::Real beta = 1.0/(2.0*std::pow(c,2));
    amrex::Real gamma = gamma_adiab-1.0;
    amrex::Real phi = gamma*ek;
  
    if(d==0){
      L_EV =  {
                {beta*(phi+c*u1),-beta*(gamma*u1+c),-beta*gamma*u2,-beta*gamma*u3,beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,2.0*beta*gamma*u3,-2.0*beta*gamma},
                {beta*(phi-c*u1),-beta*(gamma*u1-c),-beta*(gamma*u2),-beta*gamma*u3,beta*gamma},
                {u2,0.0,-1.0,0.0,0.0},
                {-u3,0.0,0.0,1.0,0.0}
              };
    }
    else if(d==1){
      L_EV =  {
                {beta*(phi+c*u2),-beta*gamma*u1,-beta*(gamma*u2+c),-beta*gamma*u3,beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,2.0*beta*gamma*u3,-2.0*beta*gamma},
                {beta*(phi-c*u2),-beta*(gamma*u1),-beta*(gamma*u2-c),-beta*gamma*u3,beta*gamma},
                {-u1,1.0,0.0,0.0,0.0},
                {u3,0.0,0.0,-1.0,0.0}
              };
    }
    else if(d==2){
      L_EV =  {
                {beta*(phi+c*u3),-beta*gamma*u1,-beta*gamma*u2,-beta*(gamma*u3+c),beta*gamma},
                {1.0-2.0*beta*phi,2.0*beta*gamma*u1,2.0*beta*gamma*u2,2.0*beta*gamma*u3,-2.0*beta*gamma},
                {beta*(phi-c*u3),-beta*(gamma*u1),-beta*gamma*u2,-beta*(gamma*u3-c),beta*gamma},
                {u1,-1.0,0.0,0.0,0.0},
                {-u2,0.0,1.0,0.0,0.0}
              };
    }
  #endif 

  return L_EV;
}

amrex::Real Compressible_Euler::pde_conservation(int lev,int d, int q,int i,int j,int k,
                                                amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                                amrex::Vector<amrex::Array4<amrex::Real>>* u) const
{
  amrex::Real cons = 0.0;
  const auto prob_lo = numerical_pde->Geom(lev).ProbLoArray();
  const auto prob_hi = numerical_pde->Geom(lev).ProbHiArray();
  const auto dx      = numerical_pde->Geom(lev).CellSizeArray();

  if((q==4 && AMREX_SPACEDIM == 2) || ((q==5 ||q==6 || q==7) && AMREX_SPACEDIM == 3)){
    //prepare data, e.g for angular momentum  
    //want to sotre in array {_idx_phi_x,idx_phi_y,idx_phi_z}, so that depending 
    //on angular momentum formulation it is easier
    //to get the correct basis function index of lienar bassi (linea in direction 
    //we want)
    
    //overall we want to map basis function index s to a 1,2,3 indexing depending 
    //on linear direction
    
    IntVect lin_phi_idx(AMREX_D_DECL(0,0,0));
    
    for(int lin_idx=0; lin_idx<AMREX_SPACEDIM; ++lin_idx){
      int s = numerical_pde->lin_mode_idx[lin_idx];

      //find linear direction
      for(int d=0; d<AMREX_SPACEDIM; ++d)
      {
        if(numerical_pde->mat_idx_s[s][d] == 1)
        {
          lin_phi_idx[d]=s;
          break;
        }
      }  
    }
    
    int l,n;
    amrex::Real vol =0.0;
    #if (AMREX_SPACEDIM == 2)
      vol = dx[0]*dx[1];
      if(q==4){n=1;l=2;}
    #elif (AMREX_SPACEDIM == 3)
      vol = dx[0]*dx[1]*dx[2];
      //cell integral angular momentum conservation
      if(q==5){n=2;l=3;}
      else if(q==6){n=3;l=1;}
      else if(q==7){n=1;l=2;}
    #endif
    
    //only for angular momentum

    amrex::Real norm_coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
    amrex::Real L_integral = ((*uw)[l])(i,j,k,n)*norm_coeff
                            *(numerical_pde->RefMat_phiphi(lin_phi_idx[n-1],lin_phi_idx[n-1], 
                            false, false)) 
                            -((*uw)[n])(i,j,k,l)*norm_coeff
                            *(numerical_pde->RefMat_phiphi(lin_phi_idx[l-1],lin_phi_idx[l-1], 
                            false, false));
                           
                            
    cons = L_integral;
    
      //NB: is jsut a but elaborated formulation because I want to use the same basis function index mapping used everyhwre so I can call the RefMat function
      //    issue is that linear basis function in direction d does not have index d, hope is clear
  }
                          
  return cons;
}*/