#ifndef MODELEQUATION_H
#define MODELEQUATION_H

#include <memory>
#include <AMReX_AmrCore.H>

//class Solver;

using namespace amrex;

template <typename NumericalMethodType>
class Mesh;

template <typename EquationType>
class ModelEquation 
{
  public:   
    
    ModelEquation() = default;

    virtual ~ModelEquation() = default;

    struct ModelVarNames{
      amrex::Vector<std::string> names;
      virtual ~ModelVarNames() = default;    
    };

    virtual const ModelVarNames& getModelVarNames() const = 0;

    template <typename... Args>
    void settings(Args... args) {
        static_cast<EquationType*>(this)->settings(std::forward<Args>(args)...);
    }

    void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
      ofs = _ofs;
    }

    void init()
    {
        static_cast<EquationType*>(this)->init();
    }

    //implementation of the physical flux present in the hyperbolic PDE
    template <typename NumericalMethodType>
    amrex::Real pde_flux(int lev, int d, int q, int m, int i, int j, int k,
                            const amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                            const amrex::Vector<amrex::Real>& xi,
                            const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const;

    //derivative of the physical flux of the hyperbolic PDE
    //if we are solving a system, then we have to specify here 
    //the unique eigenvalues of Jacobian of the flux
    template <typename NumericalMethodType>
    amrex::Real pde_dflux(int lev, int d, int q, int m, int i, int j, int k,
                              const amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                              const amrex::Vector<amrex::Real>& xi,
                              const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const;
    
    template <typename NumericalMethodType>
    amrex::Real pde_source(int lev,int q, int m, int i, int j, int k,
                              const amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                              const amrex::Vector<amrex::Real>& xi,
                              const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const;

    //return characteristic speed used for CFL number and Dt computation
    virtual amrex::Real pde_cfl_lambda(int d,int m,int i, int j, int k,
                                amrex::Vector<amrex::Array4<const amrex::Real>>* u) const = 0;

    virtual amrex::Real pde_BC_gDirichlet(int d, int side, int q)  const = 0;

    virtual amrex::Real pde_BC_gNeumann(int d, int side, int q)  const = 0;

    virtual amrex::Real pde_BC_gDirichlet(int q, int dim,const amrex::IntVect& iv, 
                                          int quad_pt_idx, const int dcomp, 
                                          const int ncomp,
                                          amrex::Array4<amrex::Real> const& dest, 
                                          amrex::GeometryData const& geom, int side, 
                                          int lev,const amrex::Vector<amrex::Vector<amrex::Real>>& gbc) 
                                          const = 0;
      
  virtual amrex::Real pde_BC_gNeumann(int q, int dim, const amrex::IntVect& iv, 
                                      int quad_pt_idx, const int dcomp,
                                      const int ncomp,
                                      amrex::Array4<amrex::Real> const& dest,
                                      amrex::GeometryData const& geom, 
                                      int side, int lev,const amrex::Vector<amrex::Vector<amrex::Real>>& gbc) 
                                      const =0;

  template <typename NumericalMethodType>
  amrex::Real pde_IC(int lev, int q, int i,int j,int k,
                        const amrex::Vector<amrex::Real>& xi,
                        const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const;

  //Number of model equations in the system
  int Q_model;

  //Number of model equatons in the system that are unique
  //i.e not lienarly dependent/composed from others
  int Q_model_unique;

  //Name of simulation/problem case solved              
  std::string model_case;
  
  //Flag to indicate if source term is considered
  bool flag_source_term;    

    /*
    void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
      ofs = _ofs;
    }
                                                            

    virtual void pde_derived_qty(int lev, int q, int m, int i, int j, int k, 
                              amrex::Vector<amrex::Array4<amrex::Real>>* u,
                              amrex::Vector<amrex::Real> xi) = 0; 
                                  
    virtual void pde_BC(int lev, int dim,int side, int q,  int quad_pt_idx,
                        const amrex::IntVect& iv, const int dcomp, const int ncomp,
                        amrex::Vector<amrex::Real>* Ubc, 
                        amrex::Vector<amrex::Real>* Ubc_valid) const = 0;
                             
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
  
     */
  protected:

    std::shared_ptr<std::ofstream> ofs;
};

template <typename EquationType>
template <typename NumericalMethodType> 
amrex::Real ModelEquation<EquationType>::pde_flux(int lev, int d, int q, int m, int i, int j, int k,
                                                  const amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                                  const amrex::Vector<amrex::Real>& xi,
                                                  const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const
{
  return static_cast<const EquationType*>(this)->pde_flux(lev,d,q,m,i,j,k,u,xi,mesh);  
}

template <typename EquationType>
template <typename NumericalMethodType>
amrex::Real ModelEquation<EquationType>::pde_dflux(int lev, int d, int q, int m, int i, int j, int k,
                                                  const amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                                  const amrex::Vector<amrex::Real>& xi,
                                                  const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const
{
  return static_cast<const EquationType*>(this)->pde_dflux(lev,d,q,m,i,j,k,u,xi,mesh);    
}

template <typename EquationType>
template <typename NumericalMethodType>
amrex::Real ModelEquation<EquationType>::pde_source(int lev,int q, int m, int i, int j, int k,
                                        const amrex::Vector<amrex::Array4<const amrex::Real>>* u,
                                        const amrex::Vector<amrex::Real>& xi,
                                        const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const
{
  return static_cast<const EquationType*>(this)->pde_source(lev,q,m,i,j,k,u,xi,mesh); 
}

template <typename EquationType>
template <typename NumericalMethodType>
amrex::Real ModelEquation<EquationType>::pde_IC(int lev, int q, int i,int j,int k,
                                              const amrex::Vector<amrex::Real>& xi,
                                              const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const
{
  return static_cast<const EquationType*>(this)->pde_IC(lev,q,i,j,k,xi,mesh);
}

#endif 
