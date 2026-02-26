#ifndef AMRDG_H
#define AMRDG_H

#include <string>
#include <limits>
#include <numeric>
#include <array>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <AMReX_Interpolater.H>
#include <AMReX_BoxDomain.H>
#include <Eigen/Core>


#include "Solver.h"
#include "Mesh.h"

using namespace amrex;

/*------------------------------------------------------------------------*/
/*
VARIABLES NAMES NOTATION
q   :   variable to iterate across solution components U=[u1,...,uq,...,uQ], it 
        is used also for quadrature loops to indicate q-th quadrature point
d   :   variable to iterate across dimensions
_w  :   indicates that the data is modal
p   :   positive/plus/+, also indicates the order of DG scheme
m   :   negative/minus/- ,
bd  :   data has been evaluated at boundary location
num :   numerical
c   :   MultiFab component indexing 
n   :   used to iterate until Np
m   :   used to iterate until Mp
l   :   used to iterate until L (levels)
x   :   used to represent point in domain
xi  :   used to represent point in reference domain
OBSERVATIONS
-MFiter are done differently depending on if we use MPI or MPI+OpenMP
  if MPI: use static tiling, no parallelizatition of tile operations
  if MPI+OMP: use dynamic tiling, each tile is given to a thread and then also 
  the mesh loop is parallelized between the threads
-some functions require to pass a pointer to either U_w or H_w, this is done because
 in this way it is easier to e.g use the same functions in the context of Runge-Kutta
*/
/*------------------------------------------------------------------------*/

class AmrDG : public Solver<AmrDG>, public std::enable_shared_from_this<AmrDG>
{
  public:
    //type alias to improve readibility
    using NumericalMethodType = AmrDG;

    AmrDG()  = default;

    ~AmrDG();

    void settings(int _p, amrex::Real _T, amrex::Real _c_dt,
                  const std::string& _limiter_type = "", amrex::Real _TVB_M = 0.0,
                  const amrex::Vector<amrex::Real>& _AMR_TVB_C = {},
                  int _t_limit = -1);

    const amrex::Vector<int>& get_lin_mode_idx() const { return lin_mode_idx; }

    void init();

    void init_bc(amrex::Vector<amrex::Vector<amrex::BCRec>>& bc, int& n_comp);

    template <typename EquationType>
    void evolve(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                const std::shared_ptr<BoundaryCondition<EquationType,
                NumericalMethodType>>& bdcond);

    template <typename EquationType>
    void time_integration(  const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                            const std::shared_ptr<BoundaryCondition<EquationType,
                            NumericalMethodType>>& bdcond,
                            amrex::Real time);

    template <typename EquationType>
    void ADER(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
              const std::shared_ptr<BoundaryCondition<EquationType,
              NumericalMethodType>>& bdcond,
              amrex::Real time);

    template <typename EquationType>
    void set_Dt(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);

    //void AMR_avg_down_initial_condition();

    void AMR_advanced_settings();

    void AMR_interpolate_initial_condition(int lev);

    void AMR_sync_initial_condition();

    void AMR_average_fine_coarse();

    void AMR_clear_level_data(int lev);

    void AMR_tag_cell_refinement(int lev, amrex::TagBoxArray& tags, 
                                amrex::Real time, int ngrow);

    void AMR_remake_level(int lev, amrex::Real time, const amrex::BoxArray& ba,
                          const amrex::DistributionMapping& dm);

    void AMR_make_new_fine_level(int lev, amrex::Real time,
                                const amrex::BoxArray& ba, 
                                const amrex::DistributionMapping& dm);

    void AMR_FillFromCoarsePatch (int lev, Real time, amrex::MultiFab* fmf,
                              int icomp,int ncomp);

    void AMR_FillPatch(int lev, Real time, amrex::MultiFab* mf,
                      int icomp, int ncomp);

    void AMR_set_flux_registers();

    void AMR_flux_correction();

    void set_init_data_system(int lev,const BoxArray& ba,
                              const DistributionMapping& dm);

    void set_init_data_component(int lev,const BoxArray& ba,
                                const DistributionMapping& dm, int q);

    template <typename EquationType> 
    void set_initial_condition(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, int lev);

    template <typename EquationType> 
    amrex::Real set_initial_condition_U_w(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                      int lev,int q,int n,int i,int j,int k);

    template <typename EquationType> 
    amrex::Real set_initial_condition_U(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                        int lev,int q,int i,int j,int k,
                                        const amrex::Vector<amrex::Real>& xi) const;

    void get_U_from_U_w(int M, int N, amrex::MultiFab* _U,
                        amrex::MultiFab* _U_w,
                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    template <typename EquationType>
    void source(int lev, int M,
                const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                amrex::MultiFab* _U,
                amrex::MultiFab* _S,
                const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    template <typename EquationType>
    void flux(int lev, int d, int M,
              const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
              amrex::MultiFab* _U,
              amrex::MultiFab* _F,
              const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    template <typename EquationType>
    void flux_bd(int lev, int d, int M,
                 const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                 amrex::MultiFab* _U,
                 amrex::MultiFab* _F,
                 amrex::MultiFab* _DF,
                 const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    void numflux(int lev,int d,int M, int N,
                 amrex::MultiFab* _U_m,
                 amrex::MultiFab* _U_p,
                 amrex::MultiFab* _F_m,
                 amrex::MultiFab* _F_p,
                 amrex::MultiFab* _DF_m,
                 amrex::MultiFab* _DF_p);

    amrex::Real LLF_numflux(int d, int m,int i, int j, int k, 
                amrex::Array4<const amrex::Real> up, 
                amrex::Array4<const amrex::Real> um, 
                amrex::Array4<const amrex::Real> fp,
                amrex::Array4<const amrex::Real> fm,  
                amrex::Array4<const amrex::Real> dfp,
                amrex::Array4<const amrex::Real> dfm);

    amrex::Real setBC(const amrex::Vector<amrex::Real>& bc, int comp,int dcomp,
                      int q, int lev);

    template <typename EquationType>
    int Limiter_w(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, int lev);

    template <typename EquationType>
    bool Limiter_linear_tvb(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                            int i, int j, int k,
                            amrex::Vector<amrex::Array4<amrex::Real>>* uw,
                            amrex::Vector<amrex::Array4<amrex::Real>>* vw,
                            int lev);

    void get_u_from_u_w(int c, int i, int j, int k,
                        amrex::Vector<amrex::Array4<amrex::Real>>* uw,
                        amrex::Vector<amrex::Array4<amrex::Real>>* u,
                        const amrex::Vector<amrex::Real>& xi);

    amrex::Real minmodB(amrex::Real a1, amrex::Real a2, amrex::Real a3,
                        bool& troubled_flag, int l, amrex::Real M) const;

    amrex::Real minmod(amrex::Real a1, amrex::Real a2, amrex::Real a3,
                       bool& troubled_flag) const;

    template<int P>
    struct Legendre {
        std::array<amrex::Real, P+1> val;
        std::array<amrex::Real, P+1> dval;
        std::array<amrex::Real, P+1> ddval;

        explicit Legendre(amrex::Real x);
    };

    template<int P, int D>
    struct MultiIndex {
        static constexpr int compute_Np() {
            long long result = 1;
            for (int i = 0; i < D; ++i) {
                result = result * (P + D - i) / (i + 1);
            }
            return static_cast<int>(result);
        }

        static constexpr int Np = compute_Np();

        struct Entry { int idx[D]; };

        static constexpr auto generate() {
            std::array<Entry, Np> result{};
            int ctr = 0;
            if constexpr (D == 1) {
                for (int ii = 0; ii <= P; ++ii) {
                    result[ctr].idx[0] = ii;
                    ctr++;
                }
            } else if constexpr (D == 2) {
                for (int ii = 0; ii <= P; ++ii) {
                    for (int jj = 0; jj <= P - ii; ++jj) {
                        result[ctr].idx[0] = ii;
                        result[ctr].idx[1] = jj;
                        ctr++;
                    }
                }
            } else if constexpr (D == 3) {
                for (int ii = 0; ii <= P; ++ii) {
                    for (int jj = 0; jj <= P - ii; ++jj) {
                        for (int kk = 0; kk <= P - ii - jj; ++kk) {
                            result[ctr].idx[0] = ii;
                            result[ctr].idx[1] = jj;
                            result[ctr].idx[2] = kk;
                            ctr++;
                        }
                    }
                }
            } else if constexpr (D == 4) {
                for (int ii = 0; ii <= P; ++ii) {
                    for (int jj = 0; jj <= P - ii; ++jj) {
                        for (int kk = 0; kk <= P - ii - jj; ++kk) {
                            for (int tt = 0; tt <= P - ii - jj - kk; ++tt) {
                                result[ctr].idx[0] = ii;
                                result[ctr].idx[1] = jj;
                                result[ctr].idx[2] = kk;
                                result[ctr].idx[3] = tt;
                                ctr++;
                            }
                        }
                    }
                }
            }
            return result;
        }

        static constexpr std::array<Entry, Np> table = generate();
    };

    template<int P>
    struct BasisLegendre {
        static constexpr int Np_s  = MultiIndex<P, AMREX_SPACEDIM>::Np;
        static constexpr int Np_st = MultiIndex<P, AMREX_SPACEDIM + 1>::Np;
        static constexpr int Np_t  = Np_st - Np_s;

        static amrex::Real phi_s(int idx, const amrex::Vector<amrex::Real>& x);
        static amrex::Real dphi_s(int idx, const amrex::Vector<amrex::Real>& x, int d);
        static amrex::Real ddphi_s(int idx, const amrex::Vector<amrex::Real>& x, int d1, int d2);
        static amrex::Real phi_t(int idx, amrex::Real tau);
        static amrex::Real dtphi_t(int idx, amrex::Real tau);
        static amrex::Real phi_st(int idx, const amrex::Vector<amrex::Real>& x);

        static std::array<amrex::Real, Np_s> phi_s_all(const amrex::Vector<amrex::Real>& x);
        static std::array<amrex::Real, Np_st> phi_st_all(const amrex::Vector<amrex::Real>& x);

        static amrex::Vector<amrex::Vector<int>> get_basis_idx_s();
        static amrex::Vector<amrex::Vector<int>> get_basis_idx_st();
        static amrex::Vector<amrex::Vector<int>> get_basis_idx_t();
    };

    template<int P>
    struct QuadratureGaussLegendre : public Quadrature {
        static constexpr int N = P + 1;
        using Table = std::array<std::array<double, N>, P+1>;

    private:
        static constexpr double cx_pi = 3.14159265358979323846;

        static constexpr double cx_abs(double x) {
            return x >= 0.0 ? x : -x;
        }

        static constexpr double cx_cos(double x) {
            double result = 1.0;
            double term = 1.0;
            for (int n = 1; n <= 30; ++n) {
                term *= -x * x / ((2*n - 1) * (2*n));
                result += term;
            }
            return result;
        }

        // P_n(x) via Bonnet recurrence
        static constexpr double cx_legendre(int n, double x) {
            if (n == 0) return 1.0;
            if (n == 1) return x;
            double p0 = 1.0, p1 = x;
            for (int k = 2; k <= n; ++k) {
                double p2 = ((2*k - 1) * x * p1 - (k - 1) * p0) / k;
                p0 = p1; p1 = p2;
            }
            return p1;
        }

        // P'_n(x) via identity P'_n = n*P_{n-1} + x*P'_{n-1}
        static constexpr double cx_dlegendre(int n, double x) {
            if (n == 0) return 0.0;
            double p0 = 1.0, p1 = x;
            double dp0 = 0.0, dp1 = 1.0;
            for (int k = 2; k <= n; ++k) {
                double p2 = ((2*k-1)*x*p1 - (k-1)*p0) / k;
                double dp2 = k * p1 + x * dp1;
                p0 = p1; p1 = p2;
                dp0 = dp1; dp1 = dp2;
            }
            return dp1;
        }

        // P''_n(x) via identity P''_n = (n+1)*P'_{n-1} + x*P''_{n-1}
        static constexpr double cx_ddlegendre(int n, double x) {
            if (n <= 1) return 0.0;
            double p0 = 1.0, p1 = x;
            double dp0 = 0.0, dp1 = 1.0;
            double ddp0 = 0.0, ddp1 = 0.0;
            for (int k = 2; k <= n; ++k) {
                double p2 = ((2*k-1)*x*p1 - (k-1)*p0) / k;
                double dp2 = k * p1 + x * dp1;
                double ddp2 = (k + 1) * dp1 + x * ddp1;
                p0 = p1; p1 = p2;
                dp0 = dp1; dp1 = dp2;
                ddp0 = ddp1; ddp1 = ddp2;
            }
            return ddp1;
        }

        // i-th GL node via Newton-Raphson on P_N(x) = 0, i=1..N/2
        static constexpr double cx_gl_node(int i) {
            double theta = cx_pi * (i - 0.25) / (N + 0.5);
            double x = cx_cos(theta);
            for (int iter = 0; iter < 30; ++iter) {
                double pval = cx_legendre(N, x);
                double dpval = cx_dlegendre(N, x);
                double dx = pval / dpval;
                x -= dx;
                if (cx_abs(dx) < 1e-16) break;
            }
            return x;
        }

        // All N GL nodes in paired ordering: [x1, -x1, x2, -x2, ..., 0]
        static constexpr std::array<double, N> compute_nodes() {
            std::array<double, N> result{};
            int idx = 0;
            for (int i = 1; i <= N/2; ++i) {
                double x = cx_gl_node(i);
                result[idx++] = x;
                result[idx++] = -x;
            }
            if (N % 2 != 0) {
                result[idx] = 0.0;
            }
            return result;
        }

        // GL weights: w_q = 2 / ((1 - x_q^2) * (P'_N(x_q))^2)
        static constexpr std::array<double, N> compute_weights() {
            std::array<double, N> result{};
            auto nd = compute_nodes();
            for (int q = 0; q < N; ++q) {
                double x = nd[q];
                double dp = cx_dlegendre(N, x);
                result[q] = 2.0 / ((1.0 - x*x) * dp * dp);
            }
            return result;
        }

        // val[k][q] = P_k(nodes[q])
        static constexpr Table compute_val_table() {
            Table result{};
            auto nd = compute_nodes();
            for (int k = 0; k <= P; ++k)
                for (int q = 0; q < N; ++q)
                    result[k][q] = cx_legendre(k, nd[q]);
            return result;
        }

        // dval[k][q] = P'_k(nodes[q])
        static constexpr Table compute_dval_table() {
            Table result{};
            auto nd = compute_nodes();
            for (int k = 0; k <= P; ++k)
                for (int q = 0; q < N; ++q)
                    result[k][q] = cx_dlegendre(k, nd[q]);
            return result;
        }

        // ddval[k][q] = P''_k(nodes[q])
        static constexpr Table compute_ddval_table() {
            Table result{};
            auto nd = compute_nodes();
            for (int k = 0; k <= P; ++k)
                for (int q = 0; q < N; ++q)
                    result[k][q] = cx_ddlegendre(k, nd[q]);
            return result;
        }

        // bd_val[k][side]: side 0 = x=-1, side 1 = x=+1
        static constexpr std::array<std::array<double, 2>, P+1> compute_bd_val() {
            std::array<std::array<double, 2>, P+1> result{};
            for (int k = 0; k <= P; ++k) {
                result[k][0] = cx_legendre(k, -1.0);
                result[k][1] = cx_legendre(k,  1.0);
            }
            return result;
        }

        static constexpr std::array<std::array<double, 2>, P+1> compute_bd_dval() {
            std::array<std::array<double, 2>, P+1> result{};
            for (int k = 0; k <= P; ++k) {
                result[k][0] = cx_dlegendre(k, -1.0);
                result[k][1] = cx_dlegendre(k,  1.0);
            }
            return result;
        }

        // shifted_lo_val[k][q] = P_k(0.5*nodes[q] - 0.5)  (maps to [-1, 0])
        static constexpr Table compute_shifted_lo_val() {
            Table result{};
            auto nd = compute_nodes();
            for (int k = 0; k <= P; ++k)
                for (int q = 0; q < N; ++q)
                    result[k][q] = cx_legendre(k, 0.5 * nd[q] - 0.5);
            return result;
        }

        // shifted_hi_val[k][q] = P_k(0.5*nodes[q] + 0.5)  (maps to [0, +1])
        static constexpr Table compute_shifted_hi_val() {
            Table result{};
            auto nd = compute_nodes();
            for (int k = 0; k <= P; ++k)
                for (int q = 0; q < N; ++q)
                    result[k][q] = cx_legendre(k, 0.5 * nd[q] + 0.5);
            return result;
        }

    public:
        static constexpr std::array<double, N> nodes = compute_nodes();
        static constexpr std::array<double, N> weights = compute_weights();

        static constexpr Table val   = compute_val_table();
        static constexpr Table dval  = compute_dval_table();
        static constexpr Table ddval = compute_ddval_table();

        static constexpr std::array<std::array<double, 2>, P+1> bd_val  = compute_bd_val();
        static constexpr std::array<std::array<double, 2>, P+1> bd_dval = compute_bd_dval();

        static constexpr Table shifted_lo_val = compute_shifted_lo_val();
        static constexpr Table shifted_hi_val = compute_shifted_hi_val();

        // Tensor-product index decomposition:
        // flat index m -> 1D node index for dimension d
        // d=0 slowest varying, d=D_total-1 fastest
        static constexpr int node_idx(int m, int d, int D_total) {
            int divisor = 1;
            for (int i = 0; i < D_total - 1 - d; ++i) divisor *= N;
            return (m / divisor) % N;
        }

        // For boundary quadrature: maps actual dim a to position among free dims
        // (excluding fixed dim d_fixed). Returns -1 for a == d_fixed.
        static constexpr int bd_free_pos(int a, int d_fixed) {
            if (a == d_fixed) return -1;
            return (a < d_fixed) ? a : a - 1;
        }

        // --- Runtime methods (definitions in AmrDG_Quadrature.cpp) ---
        void set_number_quadpoints() override;
        void set_quadpoints() override;
    };

    
    class L2ProjInterp : public AMR_Interpolation<L2ProjInterp>
    {
      public:
        L2ProjInterp() = default;

        ~L2ProjInterp() = default;

        void interp (const FArrayBox& crse,
                    int              crse_comp,
                    FArrayBox&       fine,
                    int              fine_comp,
                    int              ncomp,
                    const Box&       fine_region,
                    const IntVect&   ratio,
                    const Geometry&  crse_geom,
                    const Geometry&  fine_geom,
                    Vector<BCRec> const& bcr,
                    int              actual_comp,
                    int              actual_state,
                    RunOn            runon);

        void amr_scatter(int i, int j, int k, Array4<Real> const& fine, 
                                            int fcomp, Array4<Real const> const& crse, 
                                            int ccomp, int ncomp, IntVect const& ratio) noexcept;
                                            
        void average_down(const MultiFab& S_fine, int fine_comp, MultiFab& S_crse, 
                          int crse_comp, int ncomp, const IntVect& ratio, 
                          const int lev_fine, const int lev_coarse) noexcept;
        
        //AMR gater MUST be called from average down
        void amr_gather(int i, int j, int k,  Array4<Real const> const& fine,int fcomp,
                        Array4<Real> const& crse, int ccomp, 
                        int ncomp, IntVect const& ratio ) noexcept;

        const Eigen::MatrixXd& get_flux_proj_mat(int d, int child_idx, int b) const ;

        void reflux(amrex::MultiFab& U_crse, const amrex::MultiFab& correction_mf,
                    int lev, const amrex::Geometry& crse_geom) noexcept;

        Box CoarseBox (const Box& fine, int ratio);

        Box CoarseBox (const Box& fine, const IntVect& ratio);

        void interp_proj_mat();

        void flux_proj_mat();

        template<int P> void _interp_proj_mat();
        template<int P> void _flux_proj_mat();

      private:

        struct IndexMap{
          int i;
          int j;
          int k;
          int fidx;
        };

        //pass fine cell index and return overlapping coarse cell index 
        //and index locating fine cell w.r.t coarse one reference frame
        IndexMap set_fine_coarse_idx_map(int i, int j, int k, const amrex::IntVect& ratio);

        //pass coarse cell index and return all fine cells indices and their
        //respective rf-element indices to lcoate them w.r.t coarse cell
        amrex::Vector<IndexMap> set_coarse_fine_idx_map(int i, int j, int k, const amrex::IntVect& ratio);

        amrex::Vector<Eigen::MatrixXd> P_cf;

        amrex::Vector<Eigen::MatrixXd> P_fc;      

        // Matrices for projecting fine fluxes onto the COARSE LOW interface (xi = -1)
        amrex::Vector<amrex::Vector<Eigen::MatrixXd>> P_flux_fc_low;

        // Matrices for projecting fine fluxes onto the COARSE HIGH interface (xi = +1)
        amrex::Vector<amrex::Vector<Eigen::MatrixXd>> P_flux_fc_high;

        Eigen::MatrixXd M;

        Eigen::MatrixXd Minv;

    };

  protected:

    int kroneckerDelta(int a, int b) const;

    amrex::Real refMat_phiphi(int j, const amrex::Vector<amrex::Vector<int>>& idx_map_j,
                              int i, const amrex::Vector<amrex::Vector<int>>& idx_map_i) const ;

    // Limiter settings
    std::string limiter_type = "";          // "" = disabled, "TVB" = TVB limiter
    amrex::Real TVB_M = 0.0;               // TVB constant M (for limiting)
    amrex::Vector<amrex::Real> AMR_TVB_C;   // per-level TVB coefficient
    int t_limit = -1;                       // apply every t_limit timesteps (<= 0 = disabled)
    amrex::Vector<int> lin_mode_idx;        // indices of linear basis functions (one per SPACEDIM)

    //L2 projection quadrature matrix
    Eigen::MatrixXd quadmat;

    amrex::Vector<Eigen::MatrixXd> quadmat_bd;

  private:

    //Vandermonde matrix for mapping modes<->quadrature points
    void set_vandermat();

    //Element Matrix and Quadrature Matrix
    void set_ref_element_matrix();

    amrex::Real refMat_phiDphi(int j, const amrex::Vector<amrex::Vector<int>>& idx_map_j,
                              int i, const amrex::Vector<amrex::Vector<int>>& idx_map_i,
                              int dim) const ;
    
    amrex::Real refMat_tphitphi(int j,int i) const;
    
    amrex::Real refMat_tphiDtphi(int j,int i) const;
      
    amrex::Real coefficient_c(int k,int l) const;     

    void set_predictor(const amrex::MultiFab* _U_w,
                       amrex::MultiFab* _H_w);

    void get_H_from_H_w(int M, int N, amrex::MultiFab* _H,
                        amrex::MultiFab* _H_w,
                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi);

    // Boundary-specific: evaluates space-time basis at boundary quad points
    // d_fixed = fixed spatial dim, side = 0 (bdm, xi=-1) or 1 (bdp, xi=+1)
    void get_H_from_H_w_bd(int M, int N, amrex::MultiFab* _H,
                           amrex::MultiFab* _H_w,
                           int d_fixed, int side);

    void update_U_w(int lev);

    void update_H_w(int lev);

    template <typename EquationType> 
    void L1Norm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);

    template <typename EquationType> 
    void L2Norm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde);   
    
    template <typename EquationType> 
    void LpNorm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                      int _p, amrex::Vector<amrex::Vector<amrex::Real>> quad_pt, int N) const;
    
    solver::Array2D<std::unique_ptr<amrex::FluxRegister>> flux_reg;

    amrex::Vector<amrex::Vector<amrex::Real>> quad_weights_st_bdm;

    amrex::Vector<amrex::Vector<amrex::Real>> quad_weights_st_bdp;

    //Vandermonde matrix
    Eigen::MatrixXd V;
    //  inverse
    Eigen::MatrixXd Vinv;

    //Mass element matrix for ADER-DG corrector
    Eigen::MatrixXd Mk_corr;

    //Stiffness element matrix for ADER-DG corrector
    amrex::Vector<Eigen::MatrixXd> Sk_corr;

    //Mass boundary element matrix for ADER-DG corrector and predictor
    amrex::Vector<Eigen::MatrixXd> Mkbdm;
    amrex::Vector<Eigen::MatrixXd> Mkbdp;

    //Mass element matrix for source term (corrector step)
    Eigen::MatrixXd Mk_corr_src;

    //Mass element matrix for ADER predictor
    Eigen::MatrixXd Mk_h_w;
    Eigen::MatrixXd Mk_h_w_inv;
    Eigen::MatrixXd Mk_pred;

    //Stiffness element matrix for ADER predictor
    amrex::Vector<Eigen::MatrixXd> Sk_pred;
    amrex::Vector<Eigen::MatrixXd> Sk_predVinv;

    //Mass element matrix for source term (predictor step)
    Eigen::MatrixXd Mk_pred_src;
    Eigen::MatrixXd Mk_pred_srcVinv; 

    //RHS temporary for ADER-DG corrector (Np_s components, one per level)
    amrex::Vector<amrex::MultiFab> rhs_corr;

    //RHS temporary for ADER predictor (Np_st components, one per level)
    amrex::Vector<amrex::MultiFab> rhs_pred;

    //ADER predictor vector U(x,t) — (lev,q)
    solver::Array2D<amrex::MultiFab> H;

    //ADER Modal/Nodal predictor vector H_w — (lev,q)
    solver::Array2D<amrex::MultiFab> H_w;

    //ADER predictor vector U(x,t) evaluated at boundary plus (+) b+ — (lev,q)
    solver::Array2D<amrex::MultiFab> H_p;

    //ADER predictor vector U(x,t) evaluated at boundary minus (-) b- — (lev,q)
    solver::Array2D<amrex::MultiFab> H_m;

    //TODO: mybe nested functions ptr dont need to be shared
    //      also mabye can use again CRTP and define them genrally inside Solver

    // Runtime basis data (populated from BasisLegendre<P> at init)
    int Np_s;
    int Np_st;
    amrex::Vector<amrex::Vector<int>> basis_idx_s;
    amrex::Vector<amrex::Vector<int>> basis_idx_st;
    amrex::Vector<amrex::Vector<int>> basis_idx_t;

    // Runtime dispatch helper for basis evaluation (non-hot paths)
    amrex::Real phi_s(int idx, const amrex::Vector<amrex::Real>& x) const;

    // Template implementations for dispatch
    template<int P> void _get_U_from_U_w(int M, amrex::MultiFab* _U,
                        amrex::MultiFab* _U_w,
                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi);
    template<int P> void _get_H_from_H_w(int M, amrex::MultiFab* _H,
                        amrex::MultiFab* _H_w,
                        const amrex::Vector<amrex::Vector<amrex::Real>>& xi);
    template<int P> void _get_H_from_H_w_bd(int M, amrex::MultiFab* _H,
                        amrex::MultiFab* _H_w,
                        int d_fixed, int side);
    template<int P> void _set_vandermat();
    template<int P> void _set_ref_element_matrix();

    static void NewtonRhapson(amrex::Real &x, int n);

    std::shared_ptr<Quadrature> quadrule;

    std::shared_ptr<L2ProjInterp> amr_interpolator;

    void DEBUG_print_MFab();
};

//templated methods

template <typename EquationType> 
void AmrDG::set_initial_condition(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, int lev)
{
  
  //Print(*ofs) <<"AmrDG::InitialCondition() "<<lev<<"\n";
  //applies the initial condition to all the solution components modes
  amrex::Vector<amrex::MultiFab *> state_uw(Q);

  for(int q=0; q<Q; ++q){
    state_uw[q] = &(U_w(lev,q));
  }
  
#ifdef AMREX_USE_OMP
#pragma omp parallel 
#endif
  {
    amrex::Vector< amrex::Array4<amrex::Real> > uw(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(*(state_uw[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(*(state_uw[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();  // Include ghost cells for IC

      for(int q=0 ; q<Q; ++q){
        uw[q] = state_uw[q]->array(mfi);
      }

      amrex::ParallelFor(bx,Np_s,[&] (int i, int j, int k, int n) noexcept
      {
        for(int q=0; q<Q; ++q){
          uw[q](i,j,k,n) = set_initial_condition_U_w(model_pde,lev,q,n,i, j, k);
        }
      });
    }   
  }
}

template <typename EquationType> 
amrex::Real AmrDG::set_initial_condition_U_w(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,int lev,int q,int n,int i,int j,int k)
{
  
  //project initial condition for solution to initial condition for solution modes         
  amrex::Real sum = 0.0;
  for(int m=0; m<quadrule->qMp_s; ++m) 
  {
    sum+= set_initial_condition_U(model_pde,lev,q,i,j,k,quadrule->xi_ref_quad_s[m])*quadmat(n,m);   
  }
  
  return (sum/(refMat_phiphi(n,basis_idx_s,n,basis_idx_s)));
}

template <typename EquationType> 
amrex::Real AmrDG::set_initial_condition_U(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                            int lev,int q,int i,int j,int k, 
                                            const amrex::Vector<amrex::Real>& xi) const
{
  auto _mesh = mesh.lock();

  amrex::Real u_ic;
  u_ic = model_pde->pde_IC(lev,q,i,j,k,xi,_mesh);

  return u_ic;
}

//  ComputeDt, time_integration
template <typename EquationType>
void AmrDG::evolve(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                  const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond)
{

  bool dtn_plt; bool dt_plt; int n; amrex::Real t; std::ostringstream oss;
  int last_progress = 0;  // Track last progress for bar updates
  amrex::Real t_last_regrid = 0.0; // Track last regrid time for dt_regrid
  amrex::Real t_last_plt = 0.0; // Track last plot time for dt_outplt
  int n_regrids = 0; // Total number of regrids performed
  long n_limited_total = 0; // Total number of cells limited

  auto _mesh = mesh.lock();

  //Set timestep idx and time
  if (restart_tstep > 0) {
    n = restart_tstep;
    t = restart_time;
    t_last_plt = t;
    t_last_regrid = t;
  } else {
    n = 0;
    t = 0.0;
  }

  if (restart_tstep == 0) {
    //Plot initial condition
    dtn_plt =  (dtn_outplt > 0);
    dt_plt = (dt_outplt > 0);
    if(dtn_plt || dt_plt){PlotFile(model_pde,U_w,n, t);}

    //Output t=0 norm
    L1Norm_DG_AMR(model_pde);
    L2Norm_DG_AMR(model_pde);
  }
  
  Solver<NumericalMethodType>::set_Dt(model_pde);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    if (m_bar) {
        std::cout << "\n";
        m_bar->set_option(indicators::option::PostfixText{"0% Starting simulation..."});
        m_bar->set_progress(0);
        std::cout << std::flush;
    }
  }

  //Flux registers and FillPatch temporaries are set up after the initial
  //regrid at n==0 inside the time loop, so no need to allocate them here
  //on the uniform grid that will be immediately discarded.
  solver::Array2D<amrex::MultiFab> fillpatch_mf;
  bool first_step = true;

  while(t<T)
  {  
    if (amrex::ParallelDescriptor::IOProcessor()) {
      if (m_bar) {
        oss.str("");
        oss.clear();

        // Calculate progress per-mille (0–1000) for finer granularity
        int progress = static_cast<int>((t / T) * 1000.0);
        progress = std::clamp(progress, 0, 1000);

        // Update the text to show current time vs total time
        int pct = progress / 10;
        oss << std::fixed << std::setprecision(0) << pct << "% "
          << "t = " << std::fixed << std::setprecision(4) << t
          << " / " << T
          << " | Dt = " << std::scientific << std::setprecision(2) << Dt;
        m_bar->set_option(indicators::option::PostfixText{oss.str()});

        // Use tick() to advance the bar visually
        while (last_progress < progress) {
          m_bar->tick();
          last_progress++;
        }
        std::cout << std::flush;
      }
    }

    //Remake existing levels and create new fine levels from coarse
    if ((_mesh->L > 1)) //&& !flag_analytical_ic)
    {
      bool do_regrid_dtn = (_mesh->dtn_regrid > 0) && (n % _mesh->dtn_regrid == 0);
      bool do_regrid_dt  = (_mesh->dt_regrid > 0) && (t - t_last_regrid >= _mesh->dt_regrid - 1e-12);
      bool do_regrid_init = first_step;

      if(do_regrid_init || do_regrid_dtn || do_regrid_dt){
        // Snapshot geometry before regrid
        int old_finest = _mesh->get_finest_lev();
        amrex::Vector<amrex::BoxArray> old_ba(old_finest + 1);
        for (int l = 0; l <= old_finest; ++l) {
          old_ba[l] = _mesh->get_BoxArray(l);
        }

        _mesh->regrid(0, t);
        amrex::ParallelDescriptor::Barrier();

        // Check if geometry actually changed
        bool grid_changed = (_mesh->get_finest_lev() != old_finest);
        if (!grid_changed) {
          for (int l = 0; l <= _mesh->get_finest_lev(); ++l) {
            if (_mesh->get_BoxArray(l) != old_ba[l]) {
              grid_changed = true;
              break;
            }
          }
        }

        if (grid_changed || first_step) {
          //clear old flux register
          flux_reg.clear();

          //construct new flux register on new grid
          AMR_set_flux_registers();

          //re-allocate FillPatch temporaries on new grid
          fillpatch_mf.resize(_mesh->get_finest_lev()+1, Q);
          for(int l=0; l<=_mesh->get_finest_lev(); ++l){
            for(int q=0; q<Q; ++q){
              fillpatch_mf(l,q).define(U_w(l,q).boxArray(), U_w(l,q).DistributionMap(),
                                        Np_s, _mesh->nghost);
            }
          }

          n_regrids++;
        }

        t_last_regrid = t;
      }
      first_step = false;
    }

    // Advance solution by one time-step.
    Solver<NumericalMethodType>::time_integration(model_pde,bdcond,t);

    // Apply limiter if enabled (finest level only: coarse covered cells
    // are overwritten by AMR_average_fine_coarse, and coarse uncovered
    // cells at fine-coarse boundaries would read inconsistent neighbor
    // data since averaging hasn't occurred yet)
    if (limiter_type.size() > 0 && t_limit > 0 && (n % t_limit == 0)) {
      n_limited_total += Limiter_w(model_pde, _mesh->get_finest_lev());
    }

    // Gather valid fine cell solutions U_w into valid coarse cells
    Solver<NumericalMethodType>::AMR_average_fine_coarse();   
    
    //Prepare inner ghost cell data for next time step
    //for grids at same level and fine-coarse interface
    //fine grids ghost cells inteprolated from coarse
    for(int l=0; l<=_mesh->get_finest_lev(); ++l){
      for(int q=0 ; q<Q; ++q){
        fillpatch_mf(l,q).setVal(0.0);
      }

      AMR_FillPatch(l, t, &fillpatch_mf(l,0), 0, Np_s);

      for(int q=0 ; q<Q; ++q){
        std::swap(U_w(l,q),fillpatch_mf(l,q));
      }
    }
      
    // Update timestep idx and physical time
    n+=1;
    t+=Dt;
    
    //Plotting at pre-specified times
    dtn_plt = (dtn_outplt > 0) && (n % dtn_outplt == 0);
    dt_plt  = (dt_outplt > 0) && (t - t_last_plt >= dt_outplt - 1e-12);
    if(dtn_plt){PlotFile(model_pde,U_w,n, t);}
    else if(dt_plt){PlotFile(model_pde,U_w,n, t); t_last_plt = t;}

    //Set time-step size
    Solver<NumericalMethodType>::set_Dt(model_pde);
    if(T-t<Dt){Dt = T-t;}    
  }

  amrex::ParallelDescriptor::ReduceLongSum(n_limited_total);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    if (m_bar && !m_bar->is_completed()) {
        m_bar->set_option(indicators::option::PostfixText{"100% Done"});
        m_bar->set_progress(1000);
        m_bar->mark_as_completed();
        std::cout << std::flush;
    }
    m_bar.reset();
    // Restore cursor visibility
    std::cout << "\033[?25h" << std::flush;
    std::cout << "\n";
    Print() << "Total number of time steps: " << n << "\n";
    Print() << "Total number of regrids: " << n_regrids << "\n";
    Print() << "Total number of cells limited: " << n_limited_total << "\n";
  }

  amrex::ParallelDescriptor::Barrier();

  //Output t=T norm
  L1Norm_DG_AMR(model_pde);
  L2Norm_DG_AMR(model_pde);
}

template <typename EquationType>
void AmrDG::time_integration(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                            const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                            amrex::Real time)
{
  auto _mesh = mesh.lock();

  ADER(model_pde,bdcond,time);

  // Add MPI synchronization after time integration
  amrex::ParallelDescriptor::Barrier();
}

template <typename EquationType>
void AmrDG::ADER(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, 
                 const std::shared_ptr<BoundaryCondition<EquationType,NumericalMethodType>>& bdcond,
                  amrex::Real time)
{ 
  //NB:this function  expectes the incoming MFabs (solution) internal ghost cells
  //to be already synchronized. This is ensured by startign from IC that is fully sync
  //and then after everytime-step, sync the updated data
  auto _mesh = mesh.lock();

  if ((_mesh->L > 1))
  {
    // Reset flux registers at beginning of timestep
    for (int l = 1; l <= _mesh->get_finest_lev(); ++l) {
      for(int q=0; q<Q; ++q){
        if (flux_reg(l,q)) {
            flux_reg(l,q)->setVal(0.0);
        }
      }
    }
  }

  //iterate from finest level to coarsest
  for (int l = _mesh->get_finest_lev(); l >= 0; --l){
    //apply BC
    Solver<NumericalMethodType>::FillBoundaryCells(bdcond, &U_w(l,0), l, time);

    //set predictor initial guess
    set_predictor(&U_w(l,0), &H_w(l,0));

    //iteratively find predictor
    int iter=0;
    while(iter<p)
    {
      get_H_from_H_w(quadrule->qMp_st,Np_st, &H(l,0), &H_w(l,0),quadrule->xi_ref_quad_st);
      if(model_pde->flag_source_term){
        Solver<NumericalMethodType>::source(l,quadrule->qMp_st,model_pde, &H(l,0), &S(l,0),quadrule->xi_ref_quad_st);
      }

      for(int d = 0; d<AMREX_SPACEDIM; ++d){
        Solver<NumericalMethodType>::flux(l,d,quadrule->qMp_st,model_pde, &H(l,0), &F(l,d,0),quadrule->xi_ref_quad_st);
      }

      //update predictor
      update_H_w(l);

      iter+=1;
    }

    //use found predictor to compute corrector
    get_H_from_H_w(quadrule->qMp_st,Np_st, &H(l,0), &H_w(l,0),quadrule->xi_ref_quad_st);
    if(model_pde->flag_source_term){
      Solver<NumericalMethodType>::source(l,quadrule->qMp_st,model_pde, &H(l,0), &S(l,0),quadrule->xi_ref_quad_st);
    }
    for(int d = 0; d<AMREX_SPACEDIM; ++d){

      Solver<NumericalMethodType>::flux(l,d,quadrule->qMp_st,model_pde, &H(l,0), &F(l,d,0),quadrule->xi_ref_quad_st);

      get_H_from_H_w_bd(quadrule->qMp_st_bd,Np_st, &H_m(l,0), &H_w(l,0), d, 0);
      Solver<NumericalMethodType>::flux_bd(l,d,quadrule->qMp_st_bd,model_pde, &H_m(l,0), &Fm(l,0), &DFm(l,0),quadrule->xi_ref_quad_st_bdm[d]);

      get_H_from_H_w_bd(quadrule->qMp_st_bd,Np_st, &H_p(l,0), &H_w(l,0), d, 1);
      Solver<NumericalMethodType>::flux_bd(l,d,quadrule->qMp_st_bd,model_pde, &H_p(l,0), &Fp(l,0), &DFp(l,0),quadrule->xi_ref_quad_st_bdp[d]);

      Solver<NumericalMethodType>::numflux(l,d,quadrule->qMp_st_bd,Np_s, &H_m(l,0), &H_p(l,0), &Fm(l,0), &Fp(l,0), &DFm(l,0), &DFp(l,0));

      if ((_mesh->L > 1))
      {
        for (int q = 0; q < Q; ++q) {
          if (l < _mesh->get_finest_lev() && flux_reg(l+1,0)) {
            flux_reg(l+1,q)->CrseAdd(Fnum_int_c(l,d,q), d,
                        0, 0, static_cast<int>(Np_s),
                        -1.0, _mesh->get_Geom(l));
          }

          if (l > 0 && flux_reg(l,0)) {
            flux_reg(l,q)->FineAdd(Fnum_int_f(l,d,q), d,
                          0, 0, static_cast<int>(Np_s),
                          1.0);
          }
        }
      }
    }
    //update corrector
    update_U_w(l);
  }

  if ((_mesh->L > 1))
  {
    AMR_flux_correction();
  }


}

template <typename EquationType>
void AmrDG::flux(int lev, int d, int M,
                 const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                 amrex::MultiFab* _U,
                 amrex::MultiFab* _F,
                 const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
    // Computes all Q components of the nonlinear flux at the given M interpolation/quadrature points xi
    auto _mesh = mesh.lock();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        amrex::Vector<amrex::Array4<amrex::Real>> flux(Q);
        amrex::Vector<amrex::Array4<const amrex::Real>> u(Q);

#ifdef AMREX_USE_OMP
        for (MFIter mfi(_F[0], MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
#else
        for (MFIter mfi(_F[0], true); mfi.isValid(); ++mfi)
#endif
        {
            const amrex::Box& bx = mfi.growntilebox();

            for (int q = 0; q < Q; ++q) {
                u[q] = _U[q].const_array(mfi);
                flux[q] = _F[q].array(mfi);
            }

            for (int q = 0; q < Q; ++q) {
                amrex::ParallelFor(bx, M, [&](int i, int j, int k, int m) noexcept {
                    flux[q](i, j, k, m) = model_pde->pde_flux(lev, d, q, m, i, j, k, &u, xi[m], _mesh);
                });
            }
        }
    }
}

template <typename EquationType>
void AmrDG::flux_bd(int lev,int d, int M,
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::MultiFab* _U,
                    amrex::MultiFab* _F,
                    amrex::MultiFab* _DF,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  //General function that computes all Q components of the non-linear flux at
  //the given set of M interpolation/quadrature points xi
  auto _mesh = mesh.lock();

  #ifdef AMREX_USE_OMP
  #pragma omp parallel
  #endif
  {
    amrex::Vector< amrex::Array4<amrex::Real> > flux(Q);
    amrex::Vector< amrex::Array4<amrex::Real> > dflux(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > u(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(_F[0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(_F[0],true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        u[q] = _U[q].const_array(mfi);
        flux[q] = _F[q].array(mfi);
        dflux[q] = _DF[q].array(mfi);
      }

      amrex::ParallelFor(bx, M,[&] (int i, int j, int k, int m) noexcept
      {
        for(int q=0 ; q<Q; ++q){
          (flux[q])(i,j,k,m) = model_pde->pde_flux(lev,d,q,m,i, j, k, &u, xi[m],_mesh);
          (dflux[q])(i,j,k,m) = model_pde->pde_dflux(lev,d,q,m,i, j, k, &u, xi[m],_mesh);
        }
      });
    }
  }
}

//compute minimum time step size s.t CFL condition is met
template <typename EquationType>
void AmrDG::set_Dt(const std::shared_ptr<ModelEquation<EquationType>>& model_pde)
{
  
  auto _mesh = mesh.lock();

  amrex::Vector<amrex::Real> dt_tmp(_mesh->get_finest_lev()+1);//TODO:proper access to finest_level (in Mesh)

  for (int l = 0; l <= _mesh->get_finest_lev(); ++l)
  {
    const auto dx = _mesh->get_dx(l);

    //compute average mesh size
    amrex::Real dx_avg = 0.0;
    for(int d = 0; d < AMREX_SPACEDIM; ++d){
      dx_avg+=(amrex::Real)dx[d];
    }
    dx_avg /= (amrex::Real)AMREX_SPACEDIM;

    //get solution evaluated at cells
    get_U_from_U_w(quadrule->qMp_s, Np_s, &U(l,0), &U_w(l,0), quadrule->xi_ref_quad_s);

    //vector to accumulate all the min dt of all cells of given layer computed by this rank
    amrex::Vector<amrex::Real> rank_min_dt;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  {
      amrex::Vector<amrex::Array4<const amrex::Real>> uc(Q);

    #ifdef AMREX_USE_OMP
      for (MFIter mfi(U(l,0),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
      for (MFIter mfi(U(l,0),true); mfi.isValid(); ++mfi)
    #endif
      {
        const amrex::Box& bx = mfi.tilebox();

        for(int q=0; q<Q; ++q){
          uc[q] = U(l,q).const_array(mfi);
        }

        amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
        {
          //compute max signal velocity lambda_max(for scalar case is derivative 
          //of flux, for system case is eigenvalue of flux jacobian
          amrex::Real lambda_max = 0.0;
          amrex::Vector<amrex::Real> lambda_d(AMREX_SPACEDIM);
          for(int d = 0; d < AMREX_SPACEDIM; ++d){  
            //compute use cell avg so m==0
            lambda_d[d]= model_pde->pde_cfl_lambda(d,0,i,j,k,&uc);
          }
          //find max signal speed across the dimensions
          auto lambda_max_  = std::max_element(lambda_d.begin(),lambda_d.end());
          lambda_max = static_cast<amrex::Real>(*lambda_max_);         

          //general CFL formulation
          CFL = (1.0/(2.0*(amrex::Real)p+1.0))*(1.0/(amrex::Real)AMREX_SPACEDIM);
          amrex::Real dt_cfl = CFL*(dx_avg/lambda_max);
        
        #ifdef AMREX_USE_OMP
          #pragma omp critical
        #endif
          {
            rank_min_dt.push_back(dt_cfl);
          }
        });         
      }
    }   

    //Find min dt across all cells of layer l
    amrex::Real rank_lev_dt_min= 1.0;
    if (!rank_min_dt.empty()) {
      //compute the min in this rank for this level   
      auto rank_lev_dt_min_ = std::min_element(rank_min_dt.begin(), rank_min_dt.end());

      if (rank_lev_dt_min_ != rank_min_dt.end()) {
        rank_lev_dt_min = static_cast<amrex::Real>(*rank_lev_dt_min_);
      }
    }
  
    //Find min for across MPI processes
    ParallelDescriptor::Barrier();
    ParallelDescriptor::ReduceRealMin(rank_lev_dt_min);//, dt_tmp.size());
    dt_tmp[l] = rank_lev_dt_min;
  }

  //Find min dt across all layers
  amrex::Real dt_min = 1.0;
  if (!dt_tmp.empty()) {
    //min across levels
    auto dt_min_  = std::min_element(dt_tmp.begin(),dt_tmp.end());
    
    if (dt_min_ != dt_tmp.end()) {
      dt_min = (amrex::Real)(*dt_min_);//static_cast<amrex::Real>(*dt_min_); 
    }
  }

  ParallelDescriptor::Barrier();
  ParallelDescriptor::ReduceRealMin(dt_min);
  Dt = c_dt*dt_min;
}

template <typename EquationType>
void AmrDG::source(int lev,int M,
                    const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                    amrex::MultiFab* _U,
                    amrex::MultiFab* _S,
                    const amrex::Vector<amrex::Vector<amrex::Real>>& xi)
{
  auto _mesh = mesh.lock();

  #ifdef AMREX_USE_OMP
  #pragma omp parallel
  #endif
  {
    amrex::Vector< amrex::Array4<amrex::Real> > source(Q);
    amrex::Vector< amrex::Array4< const amrex::Real> > u(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(_S[0],MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(_S[0],true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.growntilebox();

      for(int q=0 ; q<Q; ++q){
        u[q] = _U[q].const_array(mfi);
        source[q] = _S[q].array(mfi);
      }

      amrex::ParallelFor(bx,M,[&] (int i, int j, int k, int m) noexcept
      {
        for(int q=0 ; q<Q; ++q){
          (source[q])(i,j,k,m) = model_pde->pde_source(lev,q,m,i,j,k,&u,xi[m],_mesh);
        }
      });
    }
  }
}

template <typename EquationType> 
void AmrDG::LpNorm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                                  int _p, amrex::Vector<amrex::Vector<amrex::Real>> quad_pt, int N) const
{
  auto _mesh = mesh.lock();
 
  amrex::Vector<amrex::Vector<amrex::Real>> Lpnorm_multilevel(Q);
  amrex::Vector<amrex::Real> V_level;
 
  for(int l=0; l<=_mesh->get_finest_lev(); ++l)
  {
    amrex::Vector<const amrex::MultiFab *> state_u_h(Q);
    amrex::Vector< amrex::Array4<const amrex::Real>> uh(Q);  
      
    amrex::Vector<amrex::MultiFab> U_h_DG(Q);
    
    for(int q=0; q<Q;++q){
      amrex::BoxArray c_ba = U_w(l,q).boxArray();
      U_h_DG[q].define(c_ba, U_w(l,q).DistributionMap(), Np_s, _mesh->nghost);
      amrex::MultiFab::Copy(U_h_DG[q], U_w(l,q), 0, 0, Np_s, _mesh->nghost);
    }

    // Get number of cells of full level and intersection level
    amrex::BoxArray c_ba = U_w(l,0).boxArray();
    int N_full = (int)(c_ba.numPts());

    int N_overlap = 0;
    if(l != _mesh->get_finest_lev()){
      amrex::BoxArray f_ba = U_w(l+1,0).boxArray();
      amrex::BoxArray f_ba_c = f_ba.coarsen(_mesh->get_refRatio(l));
      N_overlap = (int)(f_ba_c.numPts());
    }
      
    auto dx = _mesh->get_Geom(l).CellSizeArray();  
    amrex::Real vol = 1.0;
    for(int d = 0; d < AMREX_SPACEDIM; ++d) {
      vol *= dx[d];
    }

    V_level.push_back((amrex::Real)(vol*(amrex::Real)(N_full-N_overlap)));

    // Compute Lp norm on full level
    for(int q=0; q<Q;++q){
      state_u_h[q] = &(U_h_DG[q]);
    }
      
    // Vector to accumulate all the full level norm (reduction sum of all cells norms)
    amrex::Vector<amrex::Real> Lpnorm_full(Q, 0.0);
    amrex::Vector<amrex::Vector<amrex::Real>> Lpnorm_full_tmp(Q);
      
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
      for (MFIter mfi(*(state_u_h)[0],true); mfi.isValid(); ++mfi){
        const amrex::Box& bx_tmp = mfi.tilebox();

        for(int q=0 ; q<Q; ++q){
          uh[q] = state_u_h[q]->const_array(mfi);
        }
            
        if(l != _mesh->get_finest_lev()){
          amrex::BoxArray f_ba = U_w(l+1,0).boxArray();
          amrex::BoxArray ba_c = f_ba.coarsen(_mesh->get_refRatio(l));
          const amrex::BoxList f_ba_lst(ba_c);
            
          // Get complement: boxes NOT covered by finer level
          amrex::BoxList f_ba_lst_compl = complementIn(bx_tmp, f_ba_lst);
                
          amrex::ParallelFor(bx_tmp,[&] (int i, int j, int k) noexcept
          {
            bool is_not_covered = false;
            amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            
            // Check if cell is in complement (NOT covered by finer level)
            for (const amrex::Box& bx : f_ba_lst_compl) {
              if(bx.contains(iv)) {
                is_not_covered = true;
                break;
              }          
            }
              
            // Compute norm only for cells NOT covered by finer level
            if(is_not_covered) {
              for(int q=0 ; q<Q; ++q){
                amrex::Real cell_Lpnorm = 0.0;
                amrex::Real w;
                amrex::Real f;
                  
                for (int m = 0; m < std::pow(N,AMREX_SPACEDIM); ++m){
                  // Quad weights for each quadrature point
                  w = 1.0;
                  for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
                    w *= 2.0/std::pow(std::assoc_legendre(N,1,quad_pt[m][d_]),2);
                  }

                  amrex::Real u_h = 0.0;          
                  for (int n = 0; n < Np_s; ++n){  
                    u_h += uh[q](i,j,k,n)*(phi_s(n,quad_pt[m]));
                  }
                        
                  amrex::Real u = 0.0;
                  u = set_initial_condition_U(model_pde,l,q,i,j,k, quad_pt[m]);
                    
                  f = std::pow(std::abs(u-u_h),(amrex::Real)_p);
                  cell_Lpnorm += (f*w);
                }
                amrex::Real coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
#pragma omp critical
                {
                  Lpnorm_full_tmp[q].push_back(cell_Lpnorm*coeff);  
                }
              }          
            }
          });
        }  
        else {
          // Finest level - include all cells
          amrex::ParallelFor(bx_tmp,[&] (int i, int j, int k) noexcept
          {
            for(int q=0 ; q<Q; ++q){
              amrex::Real cell_Lpnorm = 0.0;
              amrex::Real w;
              amrex::Real f;
                
              for (int m = 0; m < std::pow(N,AMREX_SPACEDIM); ++m){
                // Quad weights for each quadrature point
                w = 1.0;
                for(int d_=0; d_<AMREX_SPACEDIM; ++d_){
                  w *= 2.0/std::pow(std::assoc_legendre(N,1,quad_pt[m][d_]),2);
                }
                  
                amrex::Real u_h = 0.0;          
                for (int n = 0; n < Np_s; ++n){  
                  u_h += uh[q](i,j,k,n)*(phi_s(n,quad_pt[m]));
                }
                      
                amrex::Real u = 0.0;
                u = set_initial_condition_U(model_pde,l,q,i,j,k, quad_pt[m]);
                      
                f = std::pow(std::abs(u-u_h),(amrex::Real)_p);
                cell_Lpnorm += (f*w);
              }
              amrex::Real coeff = vol/std::pow(2.0,AMREX_SPACEDIM);
#pragma omp critical
              {
                Lpnorm_full_tmp[q].push_back(cell_Lpnorm*coeff);  
              }
            }      
          });    
        }  
      }
    }
    
    for(int q=0 ; q<Q; ++q){
      amrex::Real global_Lpnorm = 0.0;
      global_Lpnorm = std::accumulate(Lpnorm_full_tmp[q].begin(),
                                      Lpnorm_full_tmp[q].end(), 0.0);
                                      
      ParallelDescriptor::ReduceRealSum(global_Lpnorm);
      Lpnorm_full[q] = global_Lpnorm;
    } 
      
    for(int q=0 ; q<Q; ++q){
      Lpnorm_multilevel[q].push_back((amrex::Real)Lpnorm_full[q]);      
    }      
  }
    
  amrex::Real V_amr = (amrex::Real)std::accumulate(V_level.begin(),V_level.end(), 0.0);  
  for(int q=0 ; q<Q; ++q){
    amrex::Real Lpnorm = std::accumulate(Lpnorm_multilevel[q].begin(),
                                         Lpnorm_multilevel[q].end(), 0.0);
                                             
    Lpnorm = std::pow(Lpnorm/V_amr, 1.0/(amrex::Real)_p);
    Print().SetPrecision(17) << "--multilevel--" << "\n";
    Print().SetPrecision(17) << "L" << _p << " error norm:  " << Lpnorm << " | "
                           << "DG Order:  " << p+1 << " | solution component: " << q << "\n";
  }  
}

template <typename EquationType> 
void AmrDG::L1Norm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde)
{
  LpNorm_DG_AMR(model_pde,1, quadrule->xi_ref_quad_s,quadrule->qMp_1d);
}

template <typename EquationType> 
void AmrDG::L2Norm_DG_AMR(const std::shared_ptr<ModelEquation<EquationType>>& model_pde) 
{ 
  //TODO:actually could generalize it to p points
  //Generate 2*(p+1) quadrature points in 1D
  
  int N = 2*(quadrule->qMp_1d);
  amrex::Vector<amrex::Real> GLquadpts;
  amrex::Real xiq = 0.0;
  amrex::Real theta = 0.0;
  for(int i=1; i<= (int)(N/2); ++i)
  {
    theta = M_PI*(i - 0.25)/((double)N + 0.5);
    if((1<=i) && (i<= (int)((1.0/3.0)*(double)N))){
      xiq = (1-0.125*(1.0/std::pow(N,2))+0.125*(1.0/std::pow(N,3))
            -(1.0/384.0)*(1.0/std::pow(N,4))*(39.0-28.0*(1.0/std::pow(std::sin(theta),2))))
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

  //TODO: below will always be zero right?, therefore could just do GLquadpts.push_back(0.0);   
  if(N%2!=0)//if odd number, then i=1,...,N/2 will miss one value
  {
    int i = (N/2)+1;
    theta = M_PI*(i - 0.25)/((double)N + 0.5);
    xiq = (1.0-(1.0/(8.0*std::pow((double)N,2)))
          +(1.0/(8.0*std::pow((double)N,3))))*std::cos(theta);
    NewtonRhapson(xiq, N);
    GLquadpts.push_back(xiq);   
  }//TODO: dont rememebr why is it different than in AmrDG_Quadrature
  
  amrex::Vector<amrex::Vector<amrex::Real>> GLquadptsL2norm; 
  GLquadptsL2norm.resize((int)std::pow(N,AMREX_SPACEDIM),
                        amrex::Vector<amrex::Real> (AMREX_SPACEDIM));
                        
  #if (AMREX_SPACEDIM == 1)
  for(int i=0; i<N;++i)
  {
    GLquadptsL2norm[i][0]=GLquadpts[i];
  }
  #elif (AMREX_SPACEDIM == 2)
  for(int i=0; i<N;++i){
    for(int j=0; j<N;++j){
        GLquadptsL2norm[j+N*i][0]=GLquadpts[i];
        GLquadptsL2norm[j+N*i][1]=GLquadpts[j]; 
    }
  }
  #elif (AMREX_SPACEDIM == 3)
  for(int i=0; i<N;++i){
    for(int j=0; j<N;++j){
      for(int k=0; k<N;++k){
        GLquadptsL2norm[k+N*j+N*N*i][0]=GLquadpts[i];
        GLquadptsL2norm[k+N*j+N*N*i][1]=GLquadpts[j]; 
        GLquadptsL2norm[k+N*j+N*N*i][2]=GLquadpts[k]; 
      }
    }
  }
  #endif 
  
  LpNorm_DG_AMR(model_pde,2, GLquadptsL2norm,2*(quadrule->qMp_1d));
}

template <typename EquationType>
int AmrDG::Limiter_w(const std::shared_ptr<ModelEquation<EquationType>>& model_pde, int lev)
{
  auto _mesh = mesh.lock();

  //sync ghost cells so limiter stencil reads up-to-date neighbor data
  //(after time_integration, valid cells are updated but ghost cells are stale)
  for(int q=0; q<Q; ++q){
    U_w(lev,q).FillBoundary(_mesh->get_Geom(lev).periodicity());
  }

  amrex::Vector<amrex::MultiFab> V_w(Q);

  for(int q=0 ; q<Q; ++q){
    if(limiter_type == "TVB")
    {
      V_w[q].define(U_w(lev,q).boxArray(), U_w(lev,q).DistributionMap(), AMREX_SPACEDIM, _mesh->nghost);
      V_w[q].setVal(0.0);
    }
    else
    {
      V_w[q].define(U_w(lev,q).boxArray(), U_w(lev,q).DistributionMap(), Np_s, _mesh->nghost);
      V_w[q].setVal(0.0);
    }
  }

  amrex::Vector<amrex::MultiFab *> state_uw(Q);
  amrex::Vector<amrex::MultiFab *> state_vw(Q);
  amrex::Vector<amrex::MultiFab *> state_u(Q);

  for(int q=0; q<Q; ++q){
    state_uw[q]=&(U_w(lev,q));
    state_vw[q]=&(V_w[q]);
    state_u[q]=&(U(lev,q));
  }

  int limited_count = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:limited_count)
#endif
  {
    amrex::Vector<amrex::Array4<amrex::Real>> uw(Q);
    amrex::Vector<amrex::Array4<amrex::Real>> vw(Q);
    amrex::Vector<amrex::Array4<amrex::Real>> u(Q);

    #ifdef AMREX_USE_OMP
    for (MFIter mfi(*(state_uw[0]),MFItInfo().SetDynamic(true)); mfi.isValid(); ++mfi)
    #else
    for (MFIter mfi(*(state_uw[0]),true); mfi.isValid(); ++mfi)
    #endif
    {
      const amrex::Box& bx = mfi.tilebox();

      for(int q=0 ; q<Q; ++q){
        uw[q] = state_uw[q]->array(mfi);
        vw[q] = state_vw[q]->array(mfi);
        u[q]  = state_u[q]->array(mfi);
      }

      amrex::ParallelFor(bx,[&] (int i, int j, int k) noexcept
      {
        bool cell_limited = false;
        if(limiter_type == "TVB")
        {
          cell_limited = Limiter_linear_tvb(model_pde, i, j, k, &uw, &vw, lev);
        }
        if(cell_limited){ limited_count++; }

        // Rebuild derived (non-independent) solution components after limiting.
        // The TVB limiter only modifies modes of the Q_unique independent
        // components (e.g. rho, rho*u1, rho*u2, rho*e). Components q >= Q_unique
        // are algebraically derived from these (e.g. angular momentum
        // L3 = x1*rho*u2 - x2*rho*u1 in the Keplerian disc case).
        // After the independent modes have been limited, the derived components
        // become inconsistent and must be recomputed:
        //   1. Evaluate limited independent quantities at quadrature points
        //   2. Recompute derived quantities pointwise from the limited values
        //   3. L2-project the recomputed pointwise values back to modes
        // For standard Euler (Q_unique == Q) this block is skipped entirely.
        if(Q_unique != Q){
          int M = quadrule->qMp_s;
          // Step 1+2: evaluate limited solution at quad points and recompute
          // derived quantities pointwise
          for(int m = 0; m<M ; ++m){
            get_u_from_u_w(m, i, j, k, &uw, &u, quadrule->xi_ref_quad_s[m]);

            for(int q=Q_unique; q<Q; ++q){
              model_pde->pde_derived_qty(lev,q,m,i,j,k,&u,quadrule->xi_ref_quad_s[m]);
            }
          }

          // Step 3: L2-project recomputed pointwise values back to modal
          // representation for the derived components
          for(int q=Q_unique; q<Q; ++q){
            for(int n = 0; n<Np_s ; ++n){
              amrex::Real sum = 0.0;
              for(int m=0; m<quadrule->qMp_s; ++m)
              {
                sum+= (u[q])(i,j,k,m)*quadmat(n,m);
              }
              (uw[q])(i,j,k,n) = (sum/(refMat_phiphi(n,basis_idx_s,n,basis_idx_s)));
            }
          }
        }
      });
    }
  }
  //sync internal ghost cells
  for(int q=0; q<Q; ++q){
    U_w(lev,q).FillBoundary(_mesh->get_Geom(lev).periodicity());
  }
  return limited_count;
}

template <typename EquationType>
bool AmrDG::Limiter_linear_tvb(const std::shared_ptr<ModelEquation<EquationType>>& model_pde,
                              int i, int j, int k,
                              amrex::Vector<amrex::Array4<amrex::Real>>* uw,
                              amrex::Vector<amrex::Array4<amrex::Real>>* vw,
                              int lev)
{
  amrex::Vector<amrex::Vector<amrex::Real>> res_limit(AMREX_SPACEDIM,
                                              amrex::Vector<amrex::Real>(Q_unique, 0.0));

  amrex::Vector<amrex::Vector<amrex::Real>> L_EV;
  amrex::Vector<amrex::Vector<amrex::Real>> R_EV;

  bool troubled = false;

  for(int lin_idx=0; lin_idx<AMREX_SPACEDIM; ++lin_idx){
    int s = lin_mode_idx[lin_idx];
    int shift[] = {0,0,0};
    int _d = 0;

    for(int d=0; d<AMREX_SPACEDIM; ++d)
    {
      if(basis_idx_s[s][d] == 1)
      {
        shift[d] = 1;
        _d = d;
        break;
      }
    }

    amrex::Real Dm_u_avg;
    amrex::Real Dp_u_avg;
    amrex::Real D_u;

    L_EV = model_pde->pde_EV_Lmatrix(_d,0,i,j,k,uw);

    for(int q=0; q<Q_unique; ++q){

      Dm_u_avg = 0.0;
      Dp_u_avg = 0.0;
      D_u = 0.0;

      for(int _q=0; _q<Q_unique; ++_q)
      {
        Dm_u_avg += 0.5*L_EV[q][_q]*(((*uw)[_q])(i,j,k,0)-
                                          ((*uw)[_q])(i-shift[0],j-shift[1],k-shift[2],0));

        Dp_u_avg += 0.5*L_EV[q][_q]*(((*uw)[_q])(i+shift[0],j+shift[1],k+shift[2],0)
                                          -((*uw)[_q])(i,j,k,0));

        D_u +=L_EV[q][_q]*((*uw)[_q])(i,j,k,s);
      }
      bool tmp_flag =false;

      (*vw)[q](i,j,k,lin_idx)  = minmodB(D_u,Dm_u_avg,Dp_u_avg, tmp_flag, lev, TVB_M);

      troubled = (troubled || ((*vw)[q](i,j,k,lin_idx) != D_u));
    }

    R_EV = model_pde->pde_EV_Rmatrix(_d,0,i,j,k,uw);
    for (int q = 0; q < Q_unique; ++q){
      amrex::Real sum=0.0;
      for (int _q = 0; _q < Q_unique; ++_q){
        sum+=R_EV[q][_q]*((*vw)[_q])(i,j,k,lin_idx);
      }
      res_limit[lin_idx][q]=sum;
    }
  }

  if(troubled){
    for (int q = 0; q < Q_unique; ++q){
      for(int n=1; n<Np_s; ++n){
        (*uw)[q](i,j,k,n) = 0.0;
      }

      for(int lin_idx=0; lin_idx<AMREX_SPACEDIM; ++lin_idx){
        int s = lin_mode_idx[lin_idx];
        (*uw)[q](i,j,k,s)= res_limit[lin_idx][q];
      }
    }
  }
  return troubled;
}

#endif