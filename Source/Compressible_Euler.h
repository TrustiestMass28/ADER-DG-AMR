#ifndef COMPRESSIBLE_EULER_H
#define COMPRESSIBLE_EULER_H

#include "ModelEquation.h"
//#include "Solver.h"


class Compressible_Euler : public ModelEquation<Compressible_Euler>
{
  public:
    Compressible_Euler() = default;  

    ~Compressible_Euler() = default;
    
    struct VarNames : public ModelVarNames{
      VarNames(){
        names = { "mass_density", 
                  "momentum_x", 
                  "momentum_y", 
                  "energy_density", 
                  "angular_momentum_z"
                }; 
      }
    };

    const ModelVarNames& getModelVarNames() const override {
      static VarNames names; 
      return names;
    }

    void settings(std::string _euler_case);

    template <typename NumericalMethodType>
    amrex::Real pde_flux(int lev, int d, int q, int m, int i, int j, int k, 
                          amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                          const amrex::Vector<amrex::Real>& xi,
                          std::weak_ptr<Mesh<NumericalMethodType>> mesh) const;

    template <typename NumericalMethodType>
    amrex::Real pde_dflux(int lev, int d, int q, int m, int i, int j, int k, 
                          amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                          const amrex::Vector<amrex::Real>& xi,
                          std::weak_ptr<Mesh<NumericalMethodType>> mesh) const ;

    template <typename NumericalMethodType>        
    amrex::Real pde_source(int lev, int q, int m, int i, int j, int k, 
                            amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                            const amrex::Vector<amrex::Real>& xi,
                            std::weak_ptr<Mesh<NumericalMethodType>> mesh) const; 

    virtual amrex::Real pde_cfl_lambda(int d,int m,int i, int j, int k,
                                  amrex::Vector<amrex::Array4<const amrex::Real>>* u) const override;

    virtual amrex::Real pde_BC_gDirichlet(int d, int side, int q)  const override;

    virtual amrex::Real pde_BC_gNeumann(int d, int side, int q)  const override;

    virtual amrex::Real pde_BC_gDirichlet(int q, int dim,const amrex::IntVect& iv, 
                                          int quad_pt_idx, const int dcomp, 
                                          const int ncomp,
                                          amrex::Array4<amrex::Real> const& dest, 
                                          amrex::GeometryData const& geom, int side, 
                                          int lev,const amrex::Vector<amrex::Vector<amrex::Real>>& gbc) 
                                          const override {return 0.0;};

    virtual amrex::Real pde_BC_gNeumann(int q, int dim, const amrex::IntVect& iv, 
                                        int quad_pt_idx, const int dcomp,
                                        const int ncomp,
                                        amrex::Array4<amrex::Real> const& dest,
                                        amrex::GeometryData const& geom, 
                                        int side, int lev,const amrex::Vector<amrex::Vector<amrex::Real>>& gbc) 
                                        const override {return 0.0;};

    template <typename NumericalMethodType>
    amrex::Real pde_IC(int lev, int q, int i,int j,int k, 
                        const amrex::Vector<amrex::Real>& xi, 
                        std::weak_ptr<Mesh<NumericalMethodType>> mesh) const;
    
  private:

      //Flag to indicate if angular momentum is on/off
      bool flag_angular_momentum;

      double gamma_adiab;

      template <typename T>//T==amrex::Real, const amrex::Real
      amrex::Real Pressure(amrex::Vector<amrex::Array4<T>>* u, 
                            int i, int j, int k,int m) const;

      template <typename T>
      amrex::Real Soundspeed(amrex::Vector<amrex::Array4<T>>* u,
                              int i, int j, int k, int m) const;

      amrex::Real smooth_discontinuity(amrex::Real xi, amrex::Real a, 
                                        amrex::Real b, amrex::Real s) const;
};  

template <typename NumericalMethodType>
amrex::Real Compressible_Euler::pde_IC(int lev, int q, int i,int j,int k, 
                                      const amrex::Vector<amrex::Real>& xi, 
                                      std::weak_ptr<Mesh<NumericalMethodType>> mesh) const
{
  auto _mesh = mesh.lock();

  const auto prob_lo = _mesh->get_Geom(lev).ProbLoArray();
  const auto dx     = _mesh->get_Geom(lev).CellSizeArray();
  
  amrex::Real uw_ic; 
#if (AMREX_SPACEDIM == 1)
  amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];

  amrex::Real x = 0.5*dx[0]*xi[0]+xc;
#elif(AMREX_SPACEDIM ==2) 
  amrex::Real xc = prob_lo[0] + (i+0.5) * dx[0];
  amrex::Real yc = prob_lo[1] + (j+0.5) * dx[1];

  amrex::Real x = (dx[0]/2.0)*xi[0]+xc; 
  amrex::Real y = (dx[1]/2.0)*xi[1]+yc;

  if(model_case == "isentropic_vortex")
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
  else if(model_case == "isentropic_vortex_static")
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
  else if(model_case == "double_mach_reflection")
  {
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
  else if(model_case == "kelvin_helmolz_instability")
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
  else if(model_case == "richtmeyer_meshkov_instability")
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
  else if(model_case == "keplerian_disc")
  {
    //Kevin Schaal et. al., Astrophysical hydrodynamics with a 
    //high-order discontinuous Galerking scheme and adaptive mesh refinement,
    //Royal Astronomical Society (2015) ,https://doi.org/10.1093/mnras/stv1859

    amrex::Vector<amrex::Real> ctr_ptr = {AMREX_D_DECL(3.0,3.0,3.0)};
    amrex::Real x_shape_ctr = ctr_ptr[0];
    amrex::Real y_shape_ctr = ctr_ptr[1];   
  
    amrex::Real r = std::sqrt(((x-x_shape_ctr)*(x-x_shape_ctr)+(y-y_shape_ctr)
                    *(y-y_shape_ctr)));
        
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
  else if(model_case == "radial_shock_tube")
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

template <typename NumericalMethodType>
amrex::Real Compressible_Euler::pde_flux(int lev, int d, int q, int m, int i, int j, int k, 
                                        amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                        const amrex::Vector<amrex::Real>& xi,
                                        std::weak_ptr<Mesh<NumericalMethodType>> mesh) const
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
      auto _mesh = mesh.lock();

      //implementation of angular momentum conservation law
      const auto prob_lo = _mesh->get_Geom(lev).ProbLoArray();
      const auto prob_hi = _mesh->get_Geom(lev).ProbHiArray();
      const auto dx     = _mesh->get_Geom(lev).CellSizeArray();
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
    
    if(flag_angular_momentum)
    { 
      auto _mesh = mesh.lock();
      //implementation of angular momentum conservation law
      
      const auto prob_lo = _mesh->get_Geom(lev).ProbLoArray();
      const auto prob_hi = _mesh->get_Geom(lev).ProbHiArray();
      const auto dx     = _mesh->get_Geom(lev).CellSizeArray();
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

template <typename NumericalMethodType>
amrex::Real Compressible_Euler::pde_dflux(int lev, int d, int q, int m, int i, int j, int k, 
                                          amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                          const amrex::Vector<amrex::Real>& xi,
                                          std::weak_ptr<Mesh<NumericalMethodType>> mesh) const
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

template <typename NumericalMethodType>        
amrex::Real Compressible_Euler::pde_source(int lev, int q, int m, int i, int j, int k, 
                                          amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                          const amrex::Vector<amrex::Real>& xi,
                                          std::weak_ptr<Mesh<NumericalMethodType>> mesh) const
{
  auto _mesh = mesh.lock();

  amrex::Real s;
  if(model_case == "keplerian_disc")
  {
    const auto prob_lo = _mesh->get_Geom(lev).ProbLoArray();
    const auto dx     = _mesh->get_Geom(lev).CellSizeArray();
  
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

template <typename T>
amrex::Real Compressible_Euler::Pressure(amrex::Vector<amrex::Array4<T>>* u, 
                                        int i, int j, int k,int m) const
{
  amrex::Real prs =0.0;
  for(int d=0; d<AMREX_SPACEDIM; ++d)//velocity norm
  {
    prs+=(std::pow(((*u)[d+1])(i,j,k,m),2.0)/((*u)[0])(i,j,k,m));
  }
  prs*=(-0.5);

#if(AMREX_SPACEDIM ==2) 
  prs+=((*u)[3])(i,j,k,m);
#elif(AMREX_SPACEDIM ==3)
  prs+=((*u)[4])(i,j,k,m);
#endif
  
  prs*=(gamma_adiab-1.0);
  
  return prs;
}

template <typename T>
amrex::Real Compressible_Euler::Soundspeed(amrex::Vector<amrex::Array4<T>>* u,
                                          int i, int j, int k, int m) const
{
//return the pointwise value of soundspeed, i.e at quadrature/interpolation point
amrex::Real c;
amrex::Real prs=Pressure(u,i,j,k,m);

c = std::sqrt(gamma_adiab*(prs/((*u)[0])(i,j,k,m)));
return c;
}

#endif 

    /*
    virtual amrex::Real pde_CFL(int d,int m,int i, int j, int k,
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u) const override;
     
    virtual void pde_derived_qty(int lev, int q, int m, int i, int j, int k, 
                                      amrex::Vector<amrex::Array4<amrex::Real>>* u,
                                      amrex::Vector<amrex::Real> xi) override; 
    
    //R,L matrices for cell tagging    
    virtual amrex::Vector<amrex::Vector<amrex::Real>> 
    pde_EV_Rmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4<const amrex::Real>>* u) const override;

    virtual amrex::Vector<amrex::Vector<amrex::Real>> 
    pde_EV_Lmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4<const amrex::Real>>* u) const override;
                      
    //R,L matrices for limiting                                 
    virtual amrex::Vector<amrex::Vector<amrex::Real>> 
    pde_EV_Rmatrix(int d,int m, int i, int j, int k, 
                      amrex::Vector<amrex::Array4<amrex::Real>>* u) const override ;
                                                                               
    virtual amrex::Vector<amrex::Vector<amrex::Real>> 
    pde_EV_Lmatrix(int d,int m, int i, int j, int k, 
                    amrex::Vector<amrex::Array4<amrex::Real>>* u) const override ;
    
    virtual amrex::Real get_DU_from_U_w(int d, int q, int i, int j, int k,
                                         amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                         amrex::Vector<amrex::Real> xi ) const override;
                                        
    virtual amrex::Real get_D2U_from_U_w(int d1, int d2, int q, int i, int j, int k,
                                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                          amrex::Vector<amrex::Real> xi) const override;
                                            
    virtual amrex::Real pde_conservation(int lev,int d, int q,int i,int j,int k,
                                          amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                          amrex::Vector<amrex::Array4<amrex::Real>>* u) const override; 
                                                                    
    virtual void pde_BC(int lev, int dim,int side, int q,  int quad_pt_idx,
                            const IntVect& iv, const int dcomp, const int ncomp,
                            amrex::Vector<amrex::Real>* Ubc, 
                            amrex::Vector<amrex::Real>* Ubc_valid) const override;
                        
    virtual amrex::Real pde_BC_gDirichlet(int q, int dim,const IntVect& iv, 
                                          int quad_pt_idx, const int dcomp,const int ncomp,
                                          Array4<Real> const& dest, 
                                          GeometryData const& geom, int side, int lev) const override;
                                          
    virtual amrex::Real pde_BC_gNeumann(int q, int dim, const IntVect& iv, 
                                        int quad_pt_idx, const int dcomp,const int ncomp,
                                        Array4<Real> const& dest,
                                        GeometryData const& geom, int side, int lev) const override;
                                        
    virtual amrex::Real pde_BC_gDirichlet(int d, int side, int q) const override;
    
    virtual amrex::Real pde_BC_gNeumann(int d, int side, int q) const override;                                            

  private:     
    
    amrex::Real Pressure(amrex::Vector<amrex::Array4<const amrex::Real>>* u, 
                        int i, int j, int k,int m) const;
                        
    amrex::Real Soundspeed(amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                          int i, int j, int k, int m) const;
                          
    amrex::Real Pressure(amrex::Vector<amrex::Array4< amrex::Real>>* u, 
                        int i, int j, int k,int m) const;
                        
    amrex::Real Soundspeed(amrex::Vector<amrex::Array4< amrex::Real>>* u,
                          int i, int j, int k, int m) const;


    
    amrex::Real gamma_adiab;
    */