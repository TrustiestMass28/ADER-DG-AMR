#include "AmrDG.h"

using namespace amrex;

void AmrDG::AMR_clear_level_data(int lev)
{
  //Delete level data — flat vector layout
  for (int q = 0; q < Q; ++q) {
    U(lev,q).clear();
    U_w(lev,q).clear();
    U_center(lev,q).clear();
    if (flag_source_term) {
      S(lev,q).clear();
    }

    Fm(lev,q).clear();
    DFm(lev,q).clear();
    Fp(lev,q).clear();
    DFp(lev,q).clear();

    H(lev,q).clear();
    H_w(lev,q).clear();
    H_p(lev,q).clear();
    H_m(lev,q).clear();

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      F(lev,d,q).clear();
      Fnum(lev,d,q).clear();
      Fnum_int_f(lev,d,q).clear();
      Fnum_int_c(lev,d,q).clear();
    }
  }

  // RHS temporaries
  rhs_corr[lev].clear();
  rhs_pred[lev].clear();

  // Precomputed C-F interface face data
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (cf_face_b_coarse.size() > 0) {
      cf_face_b_coarse(lev,d).clear();
    }
    if (cf_face_b_fine.size() > 0) {
      cf_face_b_fine(lev,d).clear();
    }
    if (cf_face_child_idx.size() > 0) {
      cf_face_child_idx(lev,d).clear();
    }
  }
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

  amrex::Vector<amrex::MultiFab> _mf(Q);
  for(int q=0 ; q<Q; ++q){
    _mf[q].define(ba, dm, basefunc->Np_s, _mesh->nghost);
    _mf[q].setVal(0.0);
  }

  AMR_FillPatch(lev, time, _mf.data(), 0, basefunc->Np_s);
  //clear existing level MFabs defined on old ba,dm
  _mesh->ClearLevel(lev);

  //create new level MFabs defined on new ba,dm
  Solver<NumericalMethodType>::set_init_data_system(lev,ba,dm);

  //swap old solution MFab with newly created one
  for(int q=0 ; q<Q; ++q){
    std::swap(U_w(lev,q),_mf[q]);
  }
}

void AmrDG::AMR_interpolate_initial_condition(int lev)
{
  AMR_FillFromCoarsePatch(lev, 0.0, &U_w(lev,0), 0, basefunc->Np_s);
}

void AmrDG::AMR_sync_initial_condition()
{
    auto _mesh = mesh.lock();

    // With analytical IC, each level has an independent L2 projection.
    // Average fine data down to coarse for conservation consistency.
    // For projection IC this is a no-op in exact arithmetic (projection
    // from coarse then averaging back = identity), so skip it.
    if (flag_analytical_ic) {
        AMR_average_fine_coarse();
    }

    // Sync all ghost cells via FillPatch (same-level + fine-coarse interface).
    // Iterating l=0→finest ensures each level uses the already-synced coarser level.
    for (int l = 0; l <= _mesh->get_finest_lev(); ++l) {
        amrex::Vector<amrex::MultiFab> _mf(Q);
        for (int q = 0; q < Q; ++q) {
            const amrex::BoxArray& ba = U_w(l,q).boxArray();
            const amrex::DistributionMapping& dm = U_w(l,q).DistributionMap();
            _mf[q].define(ba, dm, basefunc->Np_s, _mesh->nghost);
            _mf[q].setVal(0.0);
        }
        AMR_FillPatch(l, 0.0, _mf.data(), 0, basefunc->Np_s);
        for (int q = 0; q < Q; ++q) {
            std::swap(U_w(l,q), _mf[q]);
        }
    }
}

//Make a new level using provided BoxArray and DistributionMapping and fill with 
//interpolated coarse level data.
void AmrDG::AMR_make_new_fine_level(int lev, amrex::Real time,
                                    const amrex::BoxArray& ba, 
                                    const amrex::DistributionMapping& dm)
{ 
  //create new level MFabs defined on new ba,dm

  Solver<NumericalMethodType>::set_init_data_system(lev,ba,dm);

  AMR_FillFromCoarsePatch(lev, time, &U_w(lev,0), 0, basefunc->Np_s);
}

// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
//also fills ghost cells
void AmrDG::AMR_FillFromCoarsePatch (int lev, Real time, amrex::MultiFab* fmf,
                                int icomp,int ncomp)
{
  auto _mesh = mesh.lock();

  amrex::CpuBndryFuncFab bcf(nullptr);
  auto dummy_bc = get_null_BC(ncomp);

  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> coarse_physbcf(_mesh->get_Geom(lev-1),dummy_bc,bcf);
  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fine_physbcf(_mesh->get_Geom(lev),dummy_bc,bcf);

  amrex::Interpolater* mapper= amr_interpolator.get();

  amrex::Vector<MultiFab*> cmf;
  amrex::Vector<Real> ctime;

  for(int q=0 ; q<Q; ++q){
    cmf.clear();
    ctime.clear();

    cmf.push_back(&(U_w(lev-1,q)));
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
void AmrDG::AMR_FillPatch(int lev, Real time, amrex::MultiFab* mf, int icomp, int ncomp)
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

      smf.push_back(&(U_w(lev,q)));
      stime.push_back(time);

      amrex::FillPatchSingleLevel(mf[q], time, smf, stime, 0, icomp, ncomp,_mesh->get_Geom(lev), physbcf, 0);
    }
  }
  else
  {
    amrex::Interpolater* mapper = amr_interpolator.get();

    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> coarse_physbcf(_mesh->get_Geom(lev-1),dummy_bc,bcf);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fine_physbcf(_mesh->get_Geom(lev),dummy_bc,bcf);

    amrex::Vector<MultiFab*> cmf, fmf;
    amrex::Vector<Real> ctime, ftime;

    for(int q=0 ; q<Q; ++q){
      cmf.clear(); ctime.clear();
      fmf.clear(); ftime.clear();

      cmf.push_back(&(U_w(lev-1,q)));
      ctime.push_back(time);

      fmf.push_back(&(U_w(lev,q)));
      ftime.push_back(time);

      amrex::FillPatchTwoLevels(mf[q], time, cmf, ctime, fmf, ftime,0, icomp, ncomp,
                                _mesh->get_Geom(lev-1), _mesh->get_Geom(lev),
                                coarse_physbcf, 0, fine_physbcf,
                                0, _mesh->get_refRatio(lev-1),mapper, dummy_bc, 0);
    }
  }
}

//averages cell centered data from finer cells to the respective covered coarse cell
void AmrDG::AMR_average_fine_coarse()
{  
  auto _mesh = mesh.lock();

  for (int l = _mesh->get_finest_lev(); l > 0; --l){
    for(int q=0; q<Q; ++q){
      amr_interpolator->average_down(U_w(l,q), 0, U_w(l-1,q), 0, U_w(l-1,q).nComp(),
                                    _mesh->get_refRatio(l-1), l,l-1);
    }
  }
}

void AmrDG::AMR_set_flux_registers()
{
    auto _mesh = mesh.lock();
    const int finest = _mesh->get_finest_lev();

    flux_reg.resize(finest + 1, Q);
    cf_face_b_coarse.resize(finest + 1, AMREX_SPACEDIM);
    cf_face_b_fine.resize(finest + 1, AMREX_SPACEDIM);
    cf_face_child_idx.resize(finest + 1, AMREX_SPACEDIM);

    // Lambda: fill non-periodic physical boundary ghost cells of an iMultiFab.
    // fill_val < 0 means copy from adjacent valid cell; fill_val >= 0 means constant.
    auto fill_boundary_ghosts = [&](amrex::iMultiFab& mfab, int lev, int fill_val) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (_mesh->get_Geom(lev).isPeriodic(d)) continue;
            const amrex::Box& domain = _mesh->get_Geom(lev).Domain();
            int dom_lo = domain.smallEnd(d);
            int dom_hi = domain.bigEnd(d);
            for (amrex::MFIter mfi(mfab); mfi.isValid(); ++mfi) {
                const amrex::Box& vbx = mfi.validbox();
                auto arr = mfab[mfi].array();
                if (vbx.smallEnd(d) == dom_lo) {
                    amrex::Box ghost_lo = amrex::adjCellLo(vbx, d, 1);
                    if (fill_val >= 0) {
                        int fv = fill_val;
                        amrex::ParallelFor(ghost_lo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            arr(i,j,k) = fv;
                        });
                    } else {
                        amrex::ParallelFor(ghost_lo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            amrex::IntVect iv{AMREX_D_DECL(i,j,k)};
                            amrex::IntVect iv_valid = iv; iv_valid[d] = dom_lo;
                            arr(iv) = arr(iv_valid);
                        });
                    }
                }
                if (vbx.bigEnd(d) == dom_hi) {
                    amrex::Box ghost_hi = amrex::adjCellHi(vbx, d, 1);
                    if (fill_val >= 0) {
                        int fv = fill_val;
                        amrex::ParallelFor(ghost_hi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            arr(i,j,k) = fv;
                        });
                    } else {
                        amrex::ParallelFor(ghost_hi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            amrex::IntVect iv{AMREX_D_DECL(i,j,k)};
                            amrex::IntVect iv_valid = iv; iv_valid[d] = dom_hi;
                            arr(iv) = arr(iv_valid);
                        });
                    }
                }
            }
        }
    };

    for (int lev = 0; lev <= finest; ++lev) {
        const amrex::BoxArray& ba = _mesh->get_BoxArray(lev);
        const amrex::DistributionMapping& dm = _mesh->get_DistributionMap(lev);

        // --- FLUX REGISTER (lev > 0) ---
        if (lev > 0) {
            for (int q = 0; q < Q; ++q) {
                flux_reg(lev,q) = std::make_unique<amrex::FluxRegister>(
                    ba, dm, _mesh->get_refRatio(lev-1), lev, basefunc->Np_s);
            }
        }

        // --- COARSE SIDE: build mask + face data (lev < finest) ---
        if (lev < finest) {
            // Build coarse-fine interface mask (local, destroyed after face data)
            amrex::iMultiFab cf_mask(ba, dm, 1, _mesh->nghost);
            cf_mask.setVal(0);
            amrex::iMultiFab level_mask = amrex::makeFineMask(
                ba, dm, _mesh->get_BoxArray(lev+1), _mesh->get_refRatio(lev), 0, 1);
            cf_mask.ParallelCopy(level_mask, 0, 0, 1);
            cf_mask.FillBoundary(_mesh->get_Geom(lev).periodicity());
            fill_boundary_ghosts(cf_mask, lev, -1); // copy from adjacent valid cell

            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::BoxArray face_ba = convert(ba, amrex::IntVect::TheDimensionVector(d));
                cf_face_b_coarse(lev,d).define(face_ba, dm, 1, 0);
                cf_face_b_coarse(lev,d).setVal(0);

                for (amrex::MFIter mfi(cf_face_b_coarse(lev,d)); mfi.isValid(); ++mfi) {
                    const amrex::Box& fbx = mfi.tilebox();
                    auto const& bc_arr = cf_face_b_coarse(lev,d)[mfi].array();
                    auto const& msk = cf_mask.const_array(mfi);
                    int face_lo_d = fbx.smallEnd(d);

                    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        amrex::IntVect iv_left{AMREX_D_DECL(i,j,k)};
                        iv_left[d] -= 1;
                        amrex::IntVect iv_right{AMREX_D_DECL(i,j,k)};

                        // Ownership rule: skip lo face of tile (previous tile's hi face)
                        if (iv_left[d] >= face_lo_d) {
                            int ml = msk(iv_left);
                            int mr = msk(iv_right);
                            if (ml != mr) {
                                bc_arr(i,j,k) = (ml == 0) ? 1 : -1;
                            }
                        }
                    });
                }
            }
        } else {
            // Finest level: no coarse-side C-F faces
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::BoxArray face_ba = convert(ba, amrex::IntVect::TheDimensionVector(d));
                cf_face_b_coarse(lev,d).define(face_ba, dm, 1, 0);
                cf_face_b_coarse(lev,d).setVal(0);
            }
        }

        // --- FINE SIDE: build mask + face data (lev > 0) ---
        if (lev > 0) {
            // Build valid-cell mask (local, destroyed after face data)
            amrex::iMultiFab valid_mask(ba, dm, 1, _mesh->nghost);
            valid_mask.setVal(0);
            for (amrex::MFIter mfi(valid_mask); mfi.isValid(); ++mfi) {
                valid_mask[mfi].setVal<amrex::RunOn::Host>(1, mfi.validbox(), 0, 1);
            }
            valid_mask.FillBoundary(_mesh->get_Geom(lev).periodicity());
            fill_boundary_ghosts(valid_mask, lev, 1); // constant 1

            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::BoxArray face_ba = convert(ba, amrex::IntVect::TheDimensionVector(d));
                cf_face_b_fine(lev,d).define(face_ba, dm, 1, 0);
                cf_face_b_fine(lev,d).setVal(0);
                cf_face_child_idx(lev,d).define(face_ba, dm, 1, 0);
                cf_face_child_idx(lev,d).setVal(0);

                for (amrex::MFIter mfi(cf_face_b_fine(lev,d)); mfi.isValid(); ++mfi) {
                    const amrex::Box& fbx = mfi.tilebox();
                    auto const& bf_arr = cf_face_b_fine(lev,d)[mfi].array();
                    auto const& ci_arr = cf_face_child_idx(lev,d)[mfi].array();
                    auto const& vmsk = valid_mask.const_array(mfi);

                    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        amrex::IntVect iv_left{AMREX_D_DECL(i,j,k)};
                        iv_left[d] -= 1;
                        amrex::IntVect iv_right{AMREX_D_DECL(i,j,k)};

                        int vl = vmsk(iv_left);
                        int vr = vmsk(iv_right);

                        if (vl != vr) {
                            int b = (vl == 0) ? 1 : -1;
                            bf_arr(i,j,k) = b;

                            int child_idx = 0;
                            int bit_pos = 0;
                            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                                if (dir == d) continue;
                                if (iv_right[dir] % 2 != 0) child_idx |= (1 << bit_pos);
                                bit_pos++;
                            }
                            ci_arr(i,j,k) = child_idx;
                        }
                    });
                }
            }
        } else {
            // Level 0: no fine-side F-C faces
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::BoxArray face_ba = convert(ba, amrex::IntVect::TheDimensionVector(d));
                cf_face_b_fine(lev,d).define(face_ba, dm, 1, 0);
                cf_face_b_fine(lev,d).setVal(0);
                cf_face_child_idx(lev,d).define(face_ba, dm, 1, 0);
                cf_face_child_idx(lev,d).setVal(0);
            }
        }
    }
}

void AmrDG::AMR_flux_correction()
{
    auto _mesh = mesh.lock();

    // Loop from the finest level `l` down to level 1
    for (int l = _mesh->get_finest_lev(); l > 0; --l) {
        for(int q=0; q<Q; ++q){
            if (flux_reg(l,q)) {
                amrex::MultiFab correction_mf(U_w(l-1,q).boxArray(),
                                              U_w(l-1,q).DistributionMap(),
                                              basefunc->Np_s, _mesh->nghost);
                correction_mf.setVal(0.0);

                amrex::MultiFab dummy_vol(U_w(l-1,q).boxArray(),
                                          U_w(l-1,q).DistributionMap(),
                                          1,  _mesh->nghost);
                dummy_vol.setVal(1.0);

                flux_reg(l,q)->Reflux(correction_mf, dummy_vol,
                                       1.0, 0, 0, basefunc->Np_s,
                                       _mesh->get_Geom(l-1));

                amr_interpolator->reflux(U_w(l-1,q),
                                         correction_mf,
                                         l-1,
                                         _mesh->get_Geom(l-1));
            }
        }
    }
}
