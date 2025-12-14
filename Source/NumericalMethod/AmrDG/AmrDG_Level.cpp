#include "AmrDG.h"

using namespace amrex;

void AmrDG::AMR_clear_level_data(int lev)
{
  //Delete level data

    auto clear_mfab_vec = [](auto& vec) {
        for (auto& mf : vec) {
            mf.clear();
        }
        vec.clear();
    };

    auto clear_mfab_vec2D = [&](auto& vec2D) {
        for (auto& vec : vec2D) {
            clear_mfab_vec(vec);
        }
        vec2D.clear();
    };

    auto clear_mfab_vec3D = [&](auto& vec3D) {
        for (auto& vec2D : vec3D) {
            clear_mfab_vec2D(vec2D);
        }
        vec3D.clear();
    };

    // Per-component fields
    clear_mfab_vec(U[lev]);
    clear_mfab_vec(U_w[lev]);
    clear_mfab_vec(U_center[lev]);
    if (flag_source_term) {
        clear_mfab_vec(S[lev]);
    }

    // Per-dimension per-component fields
    clear_mfab_vec2D(F[lev]);
    clear_mfab_vec2D(DF[lev]);
    clear_mfab_vec2D(Fm[lev]);
    clear_mfab_vec2D(DFm[lev]);
    clear_mfab_vec2D(Fp[lev]);
    clear_mfab_vec2D(DFp[lev]);
    clear_mfab_vec2D(Fnum[lev]);

    // H-related fields
    clear_mfab_vec(H[lev]);
    clear_mfab_vec(H_w[lev]);
    clear_mfab_vec2D(H_p[lev]);
    clear_mfab_vec2D(H_m[lev]);
}

void AmrDG::AMR_remake_level(int lev, amrex::Real time, const amrex::BoxArray& ba,
                            const amrex::DistributionMapping& dm) 
{

  //Remake level based on new geometry. Only the evolved solution vector
  //has to be preserved, i.e only U_w. The data transfer from old MFab
  //to new MFab (differend distribution mapping) of U_w is handled by 
  //FillPatch. All the other vector of MFab are cleared and re-filled
  //with new MFabs based on new geometry and distr.map

  auto _mesh = mesh.lock();

  amrex::Vector<amrex::MultiFab> _mf;
  _mf.resize(Q);
  for(int q=0 ; q<Q; ++q){
    _mf[q].define(ba, dm, basefunc->Np_s, _mesh->nghost);
    _mf[q].setVal(0.0);    
  } 
  
  AMR_FillPatch(lev, time, _mf, 0, basefunc->Np_s);
  //clear existing level MFabs defined on old ba,dm
  _mesh->ClearLevel(lev);
  //Solver<NumericalMethodType>::AMR_clear_level_data(lev);

  //create new level MFabs defined on new ba,dm
  Solver<NumericalMethodType>::set_init_data_system(lev,ba,dm);
  
  //swap old solution MFab with newly created one
  for(int q=0 ; q<Q; ++q){
    std::swap(U_w[lev][q],_mf[q]);  
  }
}

void AmrDG::AMR_interpolate_initial_condition(int lev)
{
  AMR_FillFromCoarsePatch(lev, 0.0, U_w[lev], 0, basefunc->Np_s);
}

//Make a new level using provided BoxArray and DistributionMapping and fill with 
//interpolated coarse level data.
void AmrDG::AMR_make_new_fine_level(int lev, amrex::Real time,
                                    const amrex::BoxArray& ba, 
                                    const amrex::DistributionMapping& dm)
{ 
  //create new level MFabs defined on new ba,dm

  Solver<NumericalMethodType>::set_init_data_system(lev,ba,dm); 

  AMR_FillFromCoarsePatch(lev, time, U_w[lev], 0, basefunc->Np_s);
}

// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
//also fills ghost cells
void AmrDG::AMR_FillFromCoarsePatch (int lev, Real time, amrex::Vector<amrex::MultiFab>& fmf, 
                                int icomp,int ncomp)
{   
  auto _mesh = mesh.lock();

  //NB: in theory we would need access to boundary conditions if we wanted to apply them here
  //because of code structure, we dont have access to BC object, therefore we need to create a tmp
  //BC dummy amrex::Vector<amrex::BCRec> dummy_bc. After projection and levlec reation, the BCs will be applied
  //to all levels at the beginning of the time-step
  amrex::CpuBndryFuncFab bcf(nullptr);
  auto dummy_bc = get_null_BC(ncomp);

  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> coarse_physbcf(_mesh->get_Geom(lev-1),dummy_bc,bcf);
  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fine_physbcf(_mesh->get_Geom(lev),dummy_bc,bcf);

  amrex::Interpolater* mapper= amr_interpolator.get();//&pc_interp;//

  amrex::Vector<MultiFab*> cmf;
  amrex::Vector<Real> ctime;
  
  for(int q=0 ; q<Q; ++q){   
    // Clear vectors at the start of each iteration
    cmf.clear();
    ctime.clear();

    //Store tmp data of the coarse MFab
    cmf.push_back(&(U_w[lev-1][q]));
    ctime.push_back(time);

    amrex::InterpFromCoarseLevel(fmf[q], time, *cmf[0], 0, icomp, ncomp,_mesh->get_Geom(lev-1), 
                                _mesh->get_Geom(lev), coarse_physbcf, 0, fine_physbcf, 0, 
                                _mesh->get_refRatio(lev-1),mapper, dummy_bc, 0);       
  }                         
}

//Fillpatch operations fill all cells, valid and ghost, from actual valid data at 
//that level, space-time interpolated data from the next-coarser level, 
//neighboring grids at the same level, and domain boundary conditions 
//(for examples that have non-periodic boundary conditions).
//NB: this function is used for regrid and not for timestepping
void AmrDG::AMR_FillPatch(int lev, Real time, amrex::Vector<amrex::MultiFab>& mf,int icomp, int ncomp)
{  

  auto _mesh = mesh.lock();

  amrex::CpuBndryFuncFab bcf(nullptr); 
  auto dummy_bc = get_null_BC(ncomp);
  
  if (lev == 0)
  {
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> physbcf(_mesh->get_Geom(lev),dummy_bc,bcf);

    amrex::Vector<MultiFab*> smf;
    amrex::Vector<Real> stime;

    for(int q=0 ; q<Q; ++q){  
      smf.clear();
      stime.clear();

      smf.push_back(&(U_w[lev][q]));
      stime.push_back(time);  

      amrex::FillPatchSingleLevel(mf[q], time, smf, stime, 0, icomp, ncomp,_mesh->get_Geom(lev), physbcf, 0);  
      //FillPatchSingleLevel()  :   fills a MultiFab and its ghost region at a single 
      //                            level of refinement. The routine is flexible enough 
      //                            to interpolate in time between two MultiFabs 
      //                            associated with different times
      //                            
      //                            calls also MultiFab::FillBoundary,
      //                            MultiFab::FillDomainBoundary()     
    }                
  }
  else
  { 
    amrex::Interpolater* mapper = amr_interpolator.get();//&pc_interp;//

    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> coarse_physbcf(_mesh->get_Geom(lev-1),dummy_bc,bcf);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fine_physbcf(_mesh->get_Geom(lev),dummy_bc,bcf);

    amrex::Vector<MultiFab*> cmf, fmf;
    amrex::Vector<Real> ctime, ftime;

    for(int q=0 ; q<Q; ++q){  
      cmf.clear(); ctime.clear();
      fmf.clear(); ftime.clear();

      cmf.push_back(&(U_w[lev-1][q]));
      ctime.push_back(time);

      fmf.push_back(&(U_w[lev][q]));
      ftime.push_back(time);

      amrex::FillPatchTwoLevels(mf[q], time, cmf, ctime, fmf, ftime,0, icomp, ncomp, 
                                _mesh->get_Geom(lev-1), _mesh->get_Geom(lev),
                                coarse_physbcf, 0, fine_physbcf, 
                                0, _mesh->get_refRatio(lev-1),mapper, dummy_bc, 0);
      //FillPatchTwoLevels()    :   fills a MultiFab and its ghost region at a single 
      //                            level of refinement, assuming there is an underlying 
      //                           coarse level. This routine is flexible enough to 
      //                            interpolate the coarser level in time first using 
      //                            FillPatchSingleLevel()
    }
  }
}

//averages cell centered data from finer cells to the respective covered coarse cell
void AmrDG::AMR_average_fine_coarse()
{  
  auto _mesh = mesh.lock();

  for (int l = _mesh->get_finest_lev(); l > 0; --l){  
    for(int q=0; q<Q; ++q){   
      amr_interpolator->average_down(U_w[l][q], 0,U_w[l-1][q],0,U_w[l-1][q].nComp(), 
                                    _mesh->get_refRatio(l-1), l,l-1); 
    }
  } 
}

void AmrDG::AMR_set_flux_registers()
{
  auto _mesh = mesh.lock();

  flux_reg.resize(_mesh->get_finest_lev()+1);
  //The flux register's job is to manage the flux exchange at the coarse-fine interface between two adjacent levels

  // Resizing to Q will fill this level with 'nullptr' (empty unique_ptrs)
  // This ensures flux_reg[0] exists and has the correct size, but holds no data.
  if (flux_reg.size() > 0) {
      flux_reg[0].resize(Q); 
  }

  for (int lev = 1; lev <=_mesh->get_finest_lev(); ++lev) {
    flux_reg[lev].resize(Q);
    for(int q=0; q<Q; ++q){  
      flux_reg[lev][q] = std::make_unique<amrex::FluxRegister>(
          U_w[lev][q].boxArray(),
          U_w[lev][q].DistributionMap(),
          _mesh->get_refRatio(lev-1),
          lev,
          basefunc->Np_s  // Number of components per equation
      );
    }
  }
}

void AmrDG::AMR_flux_correction()
{
  auto _mesh = mesh.lock();

  // Loop from the finest level `l` down to level 1
  for (int l = _mesh->get_finest_lev(); l > 0; --l) {

    amrex::Vector<amrex::MultiFab> correction_mf(Q);

    for(int q=0; q<Q; ++q){ 
      if (flux_reg[l][q]) {
          //  Get the flux mismatch DeltaF for each coarse cell at level l-1.
          // This MultiFab lives on the BoxArray of the coarse cells at the C/F interface.
          correction_mf[q].define(U_w[l-1][q].boxArray(), U_w[l-1][q].DistributionMap(), basefunc->Np_s, 0);
          correction_mf[q].setVal(0.0);

          //  Define a dummy "Volume" MultiFab
          // The specialized Reflux() we need requires a volume argument.
          // Since we want the raw integral \int (F_c - F_f) phi dS, 
          // we pass 1.0 so AMReX doesn't divide by cell volume yet.
          amrex::MultiFab dummy_vol(U_w[l-1][q].boxArray(), 
                                  U_w[l-1][q].DistributionMap(), 
                                  1, 0);
          dummy_vol.setVal(1.0);
          
          // 3. Directional Reflux with Parity Correction
          for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            // Iterate over every DG mode (basis function)
            for (int n = 0; n < basefunc->Np_s; ++n) {
              // Check Parity 
              // For  Basis, is phi_n(1) == -phi_n(-1)?
              // Assuming standard 1D Legendre or Tensor Product where 
              // you know the parity of mode 'n' in direction 'd'.
              // YOU MUST IMPLEMENT THIS CHECK:
              bool is_odd = false;

              // Retrieve the polynomial order of mode 'n' in the current normal direction 'd'
              int order_in_normal_dir = basefunc->basis_idx_s[n][d];

              // Check parity
              // If the order is odd (1, 3, 5...), we must flip the sign on the High side
              is_odd = (order_in_normal_dir % 2 != 0);

              amrex::Real scale_lo = 1.0;
              amrex::Real scale_high = is_odd ? -1.0 : 1.0;

              //Update Coarse Cells on the LOW Side of Interface ---
              // (These are the cells to the RIGHT of the flux face)
              // Interface is their "Minus" face. Your stored flux is Phi(-1).
              // MATCH -> Standard Update.
              flux_reg[l][q]->Reflux(correction_mf[q], dummy_vol, 
                                    amrex::Orientation(d, amrex::Orientation::low),
                                    scale_lo,      // Scale (Add flux)
                                    n, n, 1,  // src_comp, dst_comp, num_comp
                                    _mesh->get_Geom(l-1));


              //  Update Coarse Cells on the HIGH Side of Interface ---
              // (These are the cells to the LEFT of the flux face)
              // Interface is their "Plus" face. Your stored flux is Phi(-1).
              // They need Phi(1).
              // Even Mode: Phi(1) = Phi(-1). Standard Update OK.
              // Odd Mode:  Phi(1) = -Phi(-1). We need to FLIP the sign.
              flux_reg[l][q]->Reflux(correction_mf[q], dummy_vol, 
                                    amrex::Orientation(d, amrex::Orientation::high),
                                    scale_high, 
                                    n, n, 1, 
                                    _mesh->get_Geom(l-1));
          }
          }
          //flux_reg[l][q]->Reflux(correction_mf[q], 1.0, 0, 0, basefunc->Np_s, _mesh->get_Geom(l-1));

          // Call your new DG-specific reflux function to project the
          // scalar mismatch onto the basis functions and update the solution.
          amr_interpolator->reflux(&(U_w[l-1][q]),      // Coarse level solution to be corrected
                                    &(correction_mf[q]),    // The scalar mismatch ΔF
                                    l-1,             // The coarse level index
                                    _mesh->get_Geom(l-1));
      }
    }
  }
} 