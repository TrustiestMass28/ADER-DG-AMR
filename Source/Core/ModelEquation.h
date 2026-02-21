#ifndef MODELEQUATION_H
#define MODELEQUATION_H

#include <memory>
#include <AMReX_AmrCore.H>
#include <cmath>   // For std::abs()
#include <limits>  // For std::numeric_limits

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

  //Per-cell tagging criterion for AMR refinement
  template <typename NumericalMethodType>
  bool pde_tag_cell_refinement(int lev, int i, int j, int k,
                               amrex::Real time, amrex::Real amr_c_lev,
                               const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const;

  //This function is required to be implemented by the user such that
  //corrections can be made to floating point roundings which can
  //lead to unphysical results. Depending on the application
  void set_pde_numeric_limits();

  //Number of model equations in the system
  int Q_model;

  //Number of model equatons in the system that are unique
  //i.e not lienarly dependent/composed from others
  int Q_model_unique;

  //Name of simulation/problem case solved              
  std::string model_case;
  
  //Flag to indicate if source term is considered
  bool flag_source_term;    

    //R,L matrices for limiting (non-const Array4 overload)
    virtual amrex::Vector<amrex::Vector<amrex::Real>>
    pde_EV_Rmatrix(int d, int m, int i, int j, int k,
                  amrex::Vector<amrex::Array4<amrex::Real>>* u) const = 0;

    virtual amrex::Vector<amrex::Vector<amrex::Real>>
    pde_EV_Lmatrix(int d, int m, int i, int j, int k,
                  amrex::Vector<amrex::Array4<amrex::Real>>* u) const = 0;

    virtual void pde_derived_qty(int lev, int q, int m, int i, int j, int k,
                              amrex::Vector<amrex::Array4<amrex::Real>>* u,
                              amrex::Vector<amrex::Real> xi) = 0;
  protected:

    amrex::Real PDE_NUMERIC_LIMIT;

    std::shared_ptr<std::ofstream> ofs;
};

template <typename EquationType>
void ModelEquation<EquationType>::set_pde_numeric_limits()
{
  static_cast<EquationType*>(this)->set_pde_numeric_limits();  
}

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

template <typename EquationType>
template <typename NumericalMethodType>
bool ModelEquation<EquationType>::pde_tag_cell_refinement(int lev, int i, int j, int k,
                                                          amrex::Real time, amrex::Real amr_c_lev,
                                                          const std::shared_ptr<Mesh<NumericalMethodType>>& mesh) const
{
  return static_cast<const EquationType*>(this)->pde_tag_cell_refinement(lev, i, j, k, time, amr_c_lev, mesh);
}

#endif
