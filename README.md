# -------------------------
CURRENTLY UNDER MAINTENANCE.
I'M RESTRUCTURING THE CODE 
CURRENTLY WORKS ONLY ON SINGLE LAYER
AMR PART IS GETTING REWORKED
# -------------------------

# ADER-DG-AMR  

This thesis introduces a block-structured Adaptive Mesh Refinement (AMR) software framework designed for solving general multi-dimensional systems of hyperbolic partial differential equations (PDEs) with high accuracy in both space and time. The framework employs a modal Discontinuous Galerkin (DG) method combined with Arbitrary Derivative (ADER) time integration, enabling efficient and precise numerical solutions.

The implementation features a numerical solver based on the ADER-DG method with AMR, allowing dynamic mesh refinement to capture fine-scale features while optimizing computational efficiency. Additionally, the provided base classes offer flexibility for extending the framework to other numerical methods, such as Finite Volume Methods (FVM) and Finite Difference Methods (FDM).

For efficient mesh operations and distributed data management, the framework leverages AMReX, a high-performance library tailored for massively parallel, block-structured AMR applications.

---

## 🚀 **Setup**  

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
