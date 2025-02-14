#ifndef MODELEQUATION_H
#define MODELEQUATION_H

#include <memory>
#include <AMReX_AmrCore.H>

using namespace amrex;

class NumericalMethod;

class ModelEquation
{
  public:    
    
    ModelEquation() {};

    virtual ~ModelEquation() = default;
    //virtual ~ModelEquation() = default;  // Compiler generates default behavior
    
    //virtual void model_settings() = 0;

    // Set reference to NumericalMethod for communication
    void setNumericalMethod(std::shared_ptr<NumericalMethod> nm){
      numerical_pde = nm;
    }

    void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
      ofs = _ofs;
    }
       
    virtual amrex::Real pde_flux(int lev, int d, int q, int m, int i, int j, int k, 
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                amrex::Vector<amrex::Real> xi) const =0;
                          
    virtual amrex::Real pde_dflux(int lev, int d, int q, int m, int i, int j, int k, 
                                  amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                  amrex::Vector<amrex::Real> xi) const =0;
                                  
    virtual amrex::Real pde_source(int lev,int q, int m, int i, int j, int k, 
                                  amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                  amrex::Vector<amrex::Real> xi) const =0;
                                                                    
    virtual amrex::Real pde_IC(int lev, int q, int i,int j,int k,
                              amrex::Vector<amrex::Real> xi) = 0;

    virtual amrex::Real pde_CFL(int d,int m,int i, int j, int k,
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u) const = 0;

    virtual void pde_derived_qty(int lev, int q, int m, int i, int j, int k, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* u,
                              amrex::Vector<amrex::Real> xi) = 0; 
                                  
    virtual void pde_BC(int lev, int dim,int side, int q,  int quad_pt_idx,
                        const amrex::IntVect& iv, const int dcomp, const int ncomp,
                        amrex::Vector<amrex::Real>* Ubc, 
                        amrex::Vector<amrex::Real>* Ubc_valid) const = 0;
                        
    virtual amrex::Real pde_BC_gDirichlet(int q, int dim,const amrex::IntVect& iv, 
                                          int quad_pt_idx, const int dcomp, 
                                          const int ncomp,
                                          amrex::Array4<amrex::Real> const& dest, 
                                          amrex::GeometryData const& geom, int side, 
                                          int lev) const = 0;
                                          
    virtual amrex::Real pde_BC_gNeumann(int q, int dim, const amrex::IntVect& iv, 
                                        int quad_pt_idx, const int dcomp,
                                        const int ncomp,
                                        amrex::Array4<amrex::Real> const& dest,
                                        amrex::GeometryData const& geom, 
                                        int side, int lev) const =0;
     
    virtual amrex::Real pde_BC_gDirichlet(int d, int side, int q)  const = 0;
    
    virtual amrex::Real pde_BC_gNeumann(int d, int side, int q)  const = 0;

    //R,L matrices for cell tagging                      
    virtual amrex::Vector<amrex::Vector<amrex::Real>> 
    pde_EV_Rmatrix(int d,int m, int i, int j, int k, 
                  amrex::Vector<amrex::Array4<const amrex::Real>>* u) const = 0;

    virtual amrex::Vector<amrex::Vector<amrex::Real>> 
    pde_EV_Lmatrix(int d,int m, int i, int j, int k, 
                  amrex::Vector<amrex::Array4<const amrex::Real>>* u) const = 0;
                  
    //R,L matrices for limiting                                 
    virtual amrex::Vector<amrex::Vector<amrex::Real>> 
    pde_EV_Rmatrix(int d,int m, int i, int j, int k, 
                  amrex::Vector<amrex::Array4<amrex::Real>>* u) const = 0;
                                                                               
    virtual amrex::Vector<amrex::Vector<amrex::Real>> 
    pde_EV_Lmatrix(int d,int m, int i, int j, int k, 
                  amrex::Vector<amrex::Array4<amrex::Real>>* u) const = 0;
                                                                                  
    virtual amrex::Real get_DU_from_U_w(int d, int q, int i, int j, int k,
                                         amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                         amrex::Vector<amrex::Real> xi) const = 0;
                                        
    virtual amrex::Real get_D2U_from_U_w(int d1, int d2, int q, int i, int j, int k,
                                        amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                        amrex::Vector<amrex::Real> xi) const = 0;
    
    virtual amrex::Real pde_conservation(int lev,int d, int q,int i,int j,int k,
                                        amrex::Vector<amrex::Array4<const amrex::Real>>* uw,
                                        amrex::Vector<amrex::Array4<amrex::Real>>* u) const = 0;
  
    //variables which depends on the system fo PDEs solved
    //The class data members below are declared in model_settings        
    int Q_model;        //number of equations of the system
    int Q_model_unique; //number of lin-indep equations of the system which are non
                        //derivable from others, i.e number of solution unknowns
                        //which are independent/unique/not function of others (e.g 
                        //not the angular momentum)
                        
    std::string test_case;
    
    bool flag_angular_momentum;
    bool flag_source_term;    
     
  protected:

    std::shared_ptr<std::ofstream> ofs;

    std::shared_ptr<NumericalMethod> numerical_pde;

};

#endif 
