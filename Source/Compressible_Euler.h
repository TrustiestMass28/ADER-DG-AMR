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

    virtual amrex::Real pde_flux(int lev, int d, int q, int m, int i, int j, int k, 
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                const amrex::Vector<amrex::Real>& xi) const override {return 0.0;};

    virtual amrex::Real pde_dflux(int lev, int d, int q, int m, int i, int j, int k, 
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                const amrex::Vector<amrex::Real>& xi) const override {return 0.0;};
            
    virtual amrex::Real pde_source(int lev, int q, int m, int i, int j, int k, 
                                  amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                  const amrex::Vector<amrex::Real>& xi) const override {return 0.0;}; 

    virtual amrex::Real pde_cfl_lambda(int d,int m,int i, int j, int k,
                                  amrex::Vector<amrex::Array4<const amrex::Real>>* u) const override {return 0.0;};

    virtual amrex::Real pde_BC_gDirichlet(int d, int side, int q)  const override {return 0.0;};

    virtual amrex::Real pde_BC_gNeumann(int d, int side, int q)  const override {return 0.0;};

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

    virtual amrex::Real pde_IC(int lev, int q, int i,int j,int k, amrex::Vector<amrex::Real> xi) const override {return 0.0;};
    
  private:

      //Flag to indicate if angular momentum is on/off
      bool flag_angular_momentum;

      double gamma_adiab;


    /*
    Compressible_Euler(, 
                        
                        ); 
   
    ~Compressible_Euler() override {};
    
    

                                  


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

    amrex::Real smooth_discontinuity(amrex::Real xi, amrex::Real a, 
                                    amrex::Real b, amrex::Real s) const;
    
    amrex::Real gamma_adiab;
    */
};  

#endif 
