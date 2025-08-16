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
    
    flux_reg.resize(_mesh->L);
    
    for (int lev = 1; lev < _mesh->L; ++lev) {
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            U_w[lev][0].boxArray(),
            U_w[lev][0].DistributionMap(),
            _mesh->get_refRatio(lev-1),
            lev,
            basefunc->Np_s  // Number of components per equation
        );
    }
}
