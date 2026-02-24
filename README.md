
# ADER-DG-AMR

![Kelvin-Helmholtz Instability — Domain](Doc/media/kh_domain.gif)

![Kelvin-Helmholtz Instability — Detail](Doc/media/kh_detail.gif)

A block-structured Adaptive Mesh Refinement (AMR) framework for solving general multi-dimensional systems of hyperbolic partial differential equations (PDEs) with high-order accuracy in both space and time. The framework combines a modal Discontinuous Galerkin (DG) spatial discretization with Arbitrary Derivative (ADER) time integration, yielding a single-step, single-stage, fully discrete scheme of order p+1.

The base classes are designed for extensibility: new numerical methods (FVM, FDM) and new PDE systems can be added by subclassing `Solver` and `ModelEquation` respectively. All hot-path dispatch uses the Curiously Recurring Template Pattern (CRTP) for zero-overhead compile-time polymorphism.

The framework leverages [AMReX](https://github.com/AMReX-Codes/amrex) for mesh management, distributed data structures, and parallel I/O. Supports MPI, OpenMP, and hybrid MPI+OpenMP execution.

---

## Architecture

### Directory Structure

```
Source/
  main.cpp                                # Entry point and simulation configuration
  Core/
    Simulation.h                          # Top-level orchestrator (owns all shared_ptrs)
    Solver.h                              # CRTP base for numerical methods
    ModelEquation.h                       # CRTP base for PDE systems
    BoundaryCondition.h                   # BC handler (Dirichlet/Neumann/Periodic)
    Mesh.h                                # AMReX AmrCore subclass; delegates AMR callbacks to Solver
  NumericalMethod/AmrDG/
    AmrDG.h                               # ADER-DG solver: class declaration + templated methods
    AmrDG.cpp                             # Core routines: init, numflux, update_U_w, update_H_w
    AmrDG_Level.cpp                       # AMR level operations: set/clear/remake/fill/average
    AmrDG_NumericalFlux.cpp               # Local Lax-Friedrichs numerical flux
    AmrDG_Interpolater.cpp                # L2 projection interpolation: scatter, gather, reflux
    AmrDG_TagCellRef.cpp                  # Cell tagging for AMR refinement
    AmrDG_Basis.cpp                       # Legendre polynomial basis functions and index mappings
    AmrDG_Quadrature.cpp                  # Gauss-Legendre quadrature points (Newton-Raphson)
    AmrDG_ElementMatrix.cpp               # Vandermonde, mass, stiffness, and boundary matrices
    AmrDG_Limiter.cpp                     # Limiter (TVB-style, currently disabled)
  Model/CompressibleEuler/
    Compressible_Euler.h                  # Euler equations: flux, IC, source, tagging (templated)
    Compressible_Euler.cpp                # Settings, CFL speed, BCs, numeric limits
  PostProcessing/
    plotting.py                           # yt-based 2D visualization with AMR grid overlay
    convergence.py                        # L1/L2 convergence rate analysis
Exec/
  GNUmakefile                             # AMReX build system configuration
Doc/
  media/                                  # GIFs and images for documentation
Library/
  amrex/                                  # AMReX framework
  eigen-3.4.0/                            # Eigen (linear algebra for projection matrices)
  indicators/                             # Progress bar library
```

### Class Hierarchy (CRTP)

The framework is built on two CRTP hierarchies connected through template parameters:

```
Solver<NumericalMethodType>               ModelEquation<EquationType>
         |                                          |
       AmrDG                               Compressible_Euler
```

**Solver chain** -- `Solver<AmrDG>` defines the interface for time integration, data management, boundary conditions, and AMR operations. `AmrDG` provides the concrete ADER-DG implementation. Inner classes (`BasisLegendre`, `QuadratureGaussLegendre`, `L2ProjInterp`) are also CRTP-derived from abstract bases in `Solver`.

**ModelEquation chain** -- `ModelEquation<Compressible_Euler>` defines the PDE interface. `Compressible_Euler` implements the physical flux, source terms, initial conditions, CFL speeds, boundary data, and cell tagging criteria. Each PDE method is templated on `NumericalMethodType` so it can access mesh geometry without virtual dispatch:

```cpp
// In ModelEquation<EquationType> (CRTP delegation):
template <typename NumericalMethodType>
Real pde_flux(lev, d, q, m, i, j, k, u*, xi, mesh) const;

// In Compressible_Euler (concrete implementation):
template <typename NumericalMethodType>
Real pde_flux(lev, d, q, m, i, j, k, u*, xi, mesh) const { ... }
```

The same pattern applies to `pde_dflux`, `pde_source`, `pde_IC`, and `pde_tag_cell_refinement`.

**Mesh** -- `Mesh<NumericalMethodType>` inherits from `amrex::AmrCore` and bridges AMReX callbacks (`MakeNewLevelFromScratch`, `MakeNewLevelFromCoarse`, `RemakeLevel`, `ErrorEst`, `ClearLevel`) to the corresponding `Solver` methods.

**Simulation** -- `Simulation<NumericalMethodType, EquationType>` is the top-level orchestrator that owns `shared_ptr`s to `Solver`, `ModelEquation`, `Mesh`, and `BoundaryCondition`, and drives `init()` -> `evolve()`.

### ADER-DG Method

The ADER-DG scheme is a single-step, single-stage method achieving order p+1 accuracy in space and time (no multi-stage Runge-Kutta needed).

**Basis and quadrature** -- Legendre polynomial tensor product basis with total-degree truncation. Spatial modes: `Np_s = C(p+D, D)`. Space-time modes: `Np_st = C(p+D+1, D+1)`. Gauss-Legendre quadrature with `p+1` points per dimension.

**Predictor** -- Solves a local (element-wise) space-time problem via Picard iteration (p iterations). Computes space-time DG modes `H_w` from spatial modes `U_w` by iterating:

```
H_w = Mk_h_w_inv * (Mk_pred * U_w - Sk_predVinv * F(H) - source_terms)
```

**Corrector** -- Updates spatial modes using the converged predictor and numerical fluxes:

```
U_w_new = Mk_corr_inv * (Mk_corr * U_w + Sk_corr * F - Mkbd * Fnum + source)
```

**Numerical flux** -- Local Lax-Friedrichs (Rusanov): `F_num = 0.5*(F_L + F_R) - 0.5*C*(U_R - U_L)` where `C = max(|df/du|_L, |df/du|_R)`.

### AMR and Coarse-Fine Coupling

**L2 projection interpolation** -- Coarse-to-fine (scatter) and fine-to-coarse (gather) transfers use L2 projection matrices precomputed via Gauss-Legendre quadrature and Eigen SVD for mass matrix inversion. This preserves polynomial accuracy up to degree p.

**Flux registers** -- AMReX `FluxRegister` objects accumulate the mismatch between fine and coarse numerical fluxes at coarse-fine interfaces. The reflux correction applies `M^{-1} * delta_F` to coarse DG modes, maintaining conservation.

**Interface detection** -- Two cell-centered integer masks per level identify coarse-fine interfaces from both the coarse side (`coarse_fine_interface_mask`) and fine side (`fine_level_valid_mask`).

**Regridding** -- Supports both step-based (`dtn_regrid`) and time-interval-based (`dt_regrid`) regridding. Cell tagging is delegated to the model equation via `pde_tag_cell_refinement`.

### Adding a New PDE System

To add a new model equation:

1. Create a class inheriting from `ModelEquation<YourEquation>`
2. Implement the required methods:
   - `pde_flux` -- physical flux F_d for each equation component and spatial dimension
   - `pde_dflux` -- maximum eigenvalue of the flux Jacobian (for Rusanov flux)
   - `pde_source` -- source terms (optional, controlled by `flag_source_term`)
   - `pde_IC` -- initial conditions (evaluated at reference-space quadrature points)
   - `pde_cfl_lambda` -- characteristic speed for CFL/timestep computation
   - `pde_BC_gDirichlet`, `pde_BC_gNeumann` -- boundary data
   - `pde_tag_cell_refinement` -- per-cell AMR refinement criterion
   - `set_pde_numeric_limits` -- floating-point safety bounds
3. Set `Q_model` (number of equations) and `model_case` in `settings()`

### Execution Flow

```
main.cpp
  Simulation<AmrDG, Compressible_Euler> sim
  sim.setModelSettings(case)          // model_pde->settings()
  sim.setNumericalSettings(p, T, cfl) // solver->settings()
  sim.setGeometrySettings(...)        // mesh construction
  sim.run()
    solver->init(model, mesh, ...)
      AmrDG::init()                   // basis, quadrature, element matrices, projection matrices
      mesh->InitFromScratch(t=0)      // allocate levels, set IC
    solver->evolve(model, bdcond)
      while (t < T):
        [regrid if due]
        ADER(model, bdcond, t)        // predictor-corrector on all levels (finest to coarsest)
        AMR_flux_correction()         // reflux at coarse-fine interfaces
        AMR_average_fine_coarse()     // L2-gather fine -> covered coarse cells
        AMR_FillPatch(all levels)     // ghost cell synchronization
        t += Dt
```

---

## 🚀 **Setup**  

### **Setup dev environment** 

Update the package list from the repositories
```sh
sudo apt update
```

Install GCC,GPP,Make
```sh
sudo apt install build-essential
```

Install OpenMPI libraries and binaries
```sh
sudo apt install openmpi-bin libopenmpi-dev
```

Navigate to the **Library** folder:  

```sh
cd /ADER-DG-AMR/Library/
```

### **Install Eigen 3.4.0**  

```sh
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
rm eigen-3.4.0.zip
```

### **Clone AMReX**  

```sh
git clone https://github.com/AMReX-Codes/amrex.git
```


### **Clone Other UTIL Libraries**  

```sh
git clone https://github.com/p-ranav/indicators.git
```

---

## 🔧 **Compile**  

Navigate to the **Exec** folder:  

```sh
cd /ADER-DG-AMR/Exec/
```

Open `GNUmakefile` and set the desired flags (`DIM`, `USE_MPI`, etc.).  

### **Build Commands**  

- **Normal Compilation**:  

  ```sh
  make
  ```

- **Parallel Compilation** (`NCPU` = number of cores):  

  ```sh
  make -j NCPU
  ```

---

## ▶️ **Run**  

Navigate to the **Exec** folder:  

```sh
cd /ADER-DG-AMR/Exec/
```

### **Run in Serial Mode**  

```sh
./main2d.gnu.ex
```

### **Run in Parallel (MPI)**  

```sh
mpiexec -n 4 ./main2d.gnu.MPI.ex
```

---

## 📜 **Notes**  

- Ensure all dependencies are installed before compiling.  
- Modify `GNUmakefile` for custom configurations.  
- Adjust `NCPU` for optimal parallel execution.  

---
